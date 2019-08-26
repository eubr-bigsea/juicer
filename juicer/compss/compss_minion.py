# coding=utf-8

import gc
import gettext
import json
import logging.config
import multiprocessing
import signal
import sys
import time
import traceback

# noinspection PyUnresolvedReferences
import datetime

import codecs
import os
import socketio
from timeit import default_timer as timer
# noinspection PyCompatibility
from concurrent.futures import ThreadPoolExecutor
# noinspection PyCompatibility
from juicer.runner import configuration
from juicer.runner import protocol as juicer_protocol

from juicer.runner.minion_base import Minion
from juicer.compss.transpiler import COMPSsTranspiler
from juicer.util import dataframe_util
from juicer.workflow.workflow import Workflow

from juicer.compss.COMPSsEnvGenerator import *


logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.compss.compss_minion')

locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')


class COMPSsMinion(Minion):
    """
    Controls the execution of COMPSs code in Lemonade Juicer.
    """

    # max idle time allowed in seconds until this minion self termination
    IDLENESS_TIMEOUT = 600
    TIMEOUT = 'timeout'
    MSG_PROCESSED = 'message_processed'

    def __init__(self, redis_conn, workflow_id, app_id, config, lang='en'):
        """Initialize the minion."""
        Minion.__init__(self, redis_conn, workflow_id, app_id, config)

        self.terminate_proc_queue = multiprocessing.Queue()
        self.execute_process = None
        self.ping_process = None
        self.module = None

        self._state = {}
        self.config = config

        self.transpiler = COMPSsTranspiler(config)
        configuration.set_config(self.config)

        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')
        sys.path.append(self.tmp_dir)

        signal.signal(signal.SIGTERM, self._terminate)

        self.mgr = socketio.RedisManager(
            config['juicer']['servers']['redis_url'],
            'job_output')

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job_future = None

        self.compss_config = config['juicer'].get('compss', {})
        self.lib_path = self.compss_config['functions_lib']
        # self termination timeout
        self.active_messages = 0
        self.self_terminate = True
        self.juicer_listener_enabled = False
        self.current_lang = lang

    def _emit_event(self, room, namespace):
        def emit_event(name, message, status, identifier, **kwargs):
            log.debug(_('Emit %s %s %s %s'), name, message, status,
                      identifier)
            data = {'message': message, 'status': status, 'id': identifier}
            data.update(kwargs)
            self.mgr.emit(name, data=data, room=str(room), namespace=namespace)

        return emit_event

    def execute(self, q):
        """
        Starts consuming jobs that must be processed by this minion.
        """
        while q.empty():
            try:
                self._process_message_nb()
            except Exception as ee:
                tb = traceback.format_exception(*sys.exc_info())
                log.exception(_('Unhandled error (%s) \n>%s'),
                              str(ee), '>\n'.join(tb))

    def _process_message(self):
        self._process_message_nb()
        if self.job_future:
            self.job_future.result()

    def _process_message_nb(self):
        # Get next message
        msg = self.state_control.pop_app_queue(self.app_id,
                                               block=True,
                                               timeout=self.IDLENESS_TIMEOUT)

        if msg is None and self.active_messages == 0:
            self._timeout_termination()
            return

        msg_info = json.loads(msg)

        # Sanity check: this minion should not process messages from another
        # workflow/app
        assert str(msg_info['workflow_id']) == self.workflow_id, \
            'Expected workflow_id=%s, got workflow_id=%s' % (
                self.workflow_id, msg_info['workflow_id'])

        assert str(msg_info['app_id']) == self.app_id, \
            'Expected app_id=%s, got app_id=%s' % (
                self.workflow_id, msg_info['app_id'])

        # Extract the message type
        msg_type = msg_info['type']
        self._generate_output(_('Processing message %s for app %s') %
                              (msg_type, self.app_id))

        # Forward the message according to its purpose
        if msg_type == juicer_protocol.EXECUTE:
            self.active_messages += 1
            log.info('Execute message received')
            job_id = msg_info['job_id']
            workflow = msg_info['workflow']

            lang = workflow.get('locale', self.current_lang)

            self._emit_event(room=job_id, namespace='/stand')(
                name='update job',
                message=_('Running job with lang {}/{}').format(
                    lang, self.current_lang),
                status='RUNNING', identifier=job_id)

            t = gettext.translation('messages', locales_path, [lang],
                                    fallback=True)
            t.install()

            # TODO: We should consider the case in which the spark session is
            # already instantiated and this new request asks for a different set
            # of configurations:
            # - Should we rebuild the context from scratch and execute all jobs
            # so far?
            # - Should we ignore this part of the request and execute over the
            # existing
            # (old configs) spark session?
            app_configs = msg_info.get('app_configs', {})

            if self.job_future:
                self.job_future.result()

            self.job_future = self._execute_future(job_id, workflow,
                                                   app_configs)
            log.info(_('Execute message finished'))
        elif msg_type == juicer_protocol.DELIVER:
            pass
        elif msg_type == juicer_protocol.TERMINATE:
            job_id = msg_info.get('job_id', None)
            if job_id:
                log.info(_('Terminate message received (job_id=%s)'),
                         job_id)
                self.cancel_job(job_id)
            else:
                log.info(_('Terminate message received (app=%s)'),
                         self.app_id)
                self.terminate()

        elif msg_type == Minion.MSG_PROCESSED:
            self.active_messages -= 1

        else:
            log.warn(_('Unknown message type %s'), msg_type)
            self._generate_output(_('Unknown message type %s') % msg_type)

    def _execute_future(self, job_id, workflow, app_configs):
        return self.executor.submit(self._perform_execute,
                                    job_id, workflow, app_configs)

    def _perform_execute(self, job_id, workflow, app_configs):

        # Sleeps 1s in order to wait for client join notification room
        time.sleep(1)

        result = True
        start = timer()
        try:
            loader = Workflow(workflow, self.config)

            # force the compss context creation
            self.get_or_create_compss_session(loader, app_configs)

            # Mark job as running
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job', message=_('Running job'),
                status='RUNNING', identifier=job_id)

            module_name = 'juicer_app_{}_{}_{}'.format(
                self.workflow_id,
                self.app_id,
                job_id)

            generated_code_path = os.path.join(
                self.tmp_dir, '{}.py'.format(module_name))

            app_tar_path = os.path.join(
                self.tmp_dir, '{}.tar.gz'.format(module_name))

            with codecs.open(generated_code_path, 'w', 'utf8') as out:
                self.transpiler.transpile(
                    loader.workflow, loader.graph, {}, out, job_id)

            # Get rid of .pyc file if it exists
            if os.path.isfile('{}c'.format(generated_code_path)):
                os.remove('{}c'.format(generated_code_path))

            # Setting the project and resource xml files
            docker_image = self.compss_config.get('docker_image',
                                                  'lucasmsp/compssbase:2.0')
            configs_envCOMPSs = {}
            configs_envCOMPSs['MinimumVMs'] = \
                self.compss_config.get('MinimumVMs', 1)
            configs_envCOMPSs['MaximumVMs'] = \
                self.compss_config.get('MaximumVMs', 8)
            configs_envCOMPSs['image'] = docker_image
            configs_envCOMPSs['Application'] = app_tar_path
            configs_envCOMPSs['instances'] =  \
                self.compss_config.get('instances', ['small', 'medium'])

            generated_project = os.path.join(
                self.tmp_dir, '{}_project.xml'.format(module_name))
            with codecs.open(generated_project, 'w', 'utf8') as out:
                generateProject(configs_envCOMPSs, out)

            generated_resources = os.path.join(
                self.tmp_dir, '{}_resources.xml'.format(module_name))
            with codecs.open(generated_resources, 'w', 'utf8') as out:
                generateResources(configs_envCOMPSs, out)

            # Compress the files and the lib in a tar
            make_tarfile(app_tar_path,
                         generated_code_path,
                         generated_project,
                         generated_resources,
                         self.lib_path)

            # Launch the COMPSs docker container
            import subprocess
            proc = subprocess.Popen(['docker', 'run', '-id', docker_image],
                                    stdout=subprocess.PIPE)
            containerID = proc.stdout.read()[0:15]

            command = 'sh -c "/opt/COMPSs/Runtime/scripts/user/runcompss -v"'
            output_log = os.path.join(
                self.tmp_dir, '{}_output.log'.format(module_name))
            with open(output_log, "w") as output:
                proc = subprocess.Popen(['docker', 'exec', '-it',
                                         containerID, 'sh', '-c',
                                         command], stdout=output)


            # usar o runcompss

            # # Here code is loaded in Python executor
            # # self.module = importlib.import_module(module_name)
            # # self.module = imp.reload(self.module)
            # if log.isEnabledFor(logging.debug):
            #     log.debug('Objects in memory after loading module: %s',
            #               len(gc.get_objects()))

            # Starting execution. At this point, the transpiler have created a
            # module with a main function that receives a spark_session and the
            # current state (if any). We pass the current state to the execution
            # to avoid re-computing the same tasks over and over again, in case
            # of several partial workflow executions.
            # new_state = self.module.main(
            #     self.get_or_create_spark_session(loader, app_configs),
            #     self._state,
            #     self._emit_event(room=job_id, namespace='/stand'))

            end = timer()
            # Mark job as completed
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job',
                message=_('Job finished in {0:.2f}s').format(end - start),
                status='COMPLETED', identifier=job_id)

            # We update the state incrementally, i.e., new task results can be
            # overwritten but never lost.
            # self._state.update(new_state)


        except UnicodeEncodeError as ude:
            message = self.MNN006[1].format(ude)
            log.warn(_(message))
            # Mark job as failed
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job', message=message,
                status='ERROR', identifier=job_id)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False

        except ValueError as ve:
            message = _('Invalid or missing parameters: {}').format(str(ve))
            print('#' * 30)
            import traceback
            traceback.print_exc(file=sys.stdout)
            print('#' * 30)
            log.warn(message)
            if self.transpiler.current_task_id is not None:
                self._emit_event(room=job_id, namespace='/stand')(
                    name='update task', message=message,
                    status='ERROR', identifier=self.transpiler.current_task_id)
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job', message=message,
                status='ERROR', identifier=job_id)
            self._generate_output(message, 'ERROR')
            result = False

        except SyntaxError as se:
            message = self.MNN006[1].format(se)
            log.warn(message)
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job', message=message,
                status='ERROR', identifier=job_id)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False

        except Exception as ee:
            import traceback
            tb = traceback.format_exception(*sys.exc_info())
            log.exception(_('Unhandled error'))
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job', message='\n'.join(tb),
                status='ERROR', identifier=job_id)
            self._generate_output(str(ee), 'ERROR', code=1000)
            result = False

        self.message_processed('execute')

        return result

    # noinspection PyUnresolvedReferences
    def get_or_create_compss_session(self, loader, app_configs):
        """
        """
        pass

    def _send_to_output(self, data):
        self.state_control.push_app_output_queue(
            self.app_id, json.dumps(data))

    def _read_dataframe_data(self, task_id, output, port):
        success = True
        data = []
        # Last position in state is the execution time, so it should be ignored
        if port in self._state[task_id]:
            df = self._state[task_id][port]['output']
            partial_result = self._state[task_id][port]['sample']

            # In this case we already have partial data collected for the
            # particular task
            if partial_result:
                status_data = {'status': 'SUCCESS', 'code': self.MNN002[0],
                               'message': self.MNN002[1], 'output': output}
                data = [dataframe_util.convert_to_csv(r) for r in
                        partial_result]

            # In this case we do not have partial data collected for the task
            # Then we must obtain it if the 'take' operation applies
            elif df is not None and hasattr(df, 'take'):
                # Evaluating if df has method "take" allows unit testing
                # instead of testing exact pyspark.sql.dataframe.Dataframe
                # type check.
                # FIXME define as a parameter?:
                status_data = {'status': 'SUCCESS', 'code': self.MNN002[0],
                               'message': self.MNN002[1], 'output': output}
                data = df.rdd.map(dataframe_util.convert_to_csv).take(100)

            # In this case, do not make sense to request data for this
            # particular task output port
            else:
                status_data = {'status': 'ERROR', 'code': self.MNN001[0],
                               'message': self.MNN001[1]}
                success = False

        else:
            status_data = {'status': 'ERROR', 'code': self.MNN004[0],
                           'message': self.MNN004[1]}
            success = False

        return success, status_data, data

    # noinspection PyUnusedLocal
    def cancel_job(self, job_id):
        if self.job_future:
            while True:
                # if self.is_spark_session_available():
                #     self.spark_session.sparkContext.cancelAllJobs()
                # try:
                #     self.job_future.result(timeout=1)
                #     break
                # except TimeoutError as te:
                #     pass
                pass

        message = self.MNN007[1].format(self.app_id)
        log.info(message)
        self._generate_output(message, 'SUCCESS', self.MNN007[0])

    def _timeout_termination(self):
        if not self.self_terminate:
            return
        termination_msg = {
            'workflow_id': self.workflow_id,
            'app_id': self.app_id,
            'type': 'terminate'
        }
        log.info('Requesting termination (workflow_id=%s,app_id=%s) %s %s',
                 self.workflow_id, self.app_id,
                 ' due idleness timeout. Msg: ', termination_msg)
        self.state_control.push_start_queue(json.dumps(termination_msg))

    def message_processed(self, msg_type):
        msg_processed = {
            'workflow_id': self.workflow_id,
            'app_id': self.app_id,
            'type': COMPSsMinion.MSG_PROCESSED,
            'msg_type': msg_type
        }
        self.state_control.push_app_queue(self.app_id,
                                          json.dumps(msg_processed))
        log.info('Sending message processed message: %s' % msg_processed)

    # noinspection PyUnusedLocal
    def _terminate(self, _signal, _frame):
        self.terminate()

    def terminate(self):
        """
        This is a handler that reacts to a sigkill signal. The most feasible
        scenario is when the JuicerServer is demanding the termination of this
        minion. In this case, we stop and release any allocated resource
        (spark_session) and kill the subprocess managed in here.
        """
        # if self.spark_session:
        #     self.spark_session.stop()
        #     self.spark_session.sparkContext.stop()
        #     self.spark_session = None

        log.info('Post terminate message in queue')
        self.terminate_proc_queue.put({'terminate': True})

        # if self.execute_process:
        #     os.kill(self.execute_process.pid, signal.SIGKILL)
        # if self.ping_process:
        #     os.kill(self.ping_process.pid, signal.SIGKILL)

        message = self.MNN008[1].format(self.app_id)
        log.info(message)
        self._generate_output(message, 'SUCCESS', self.MNN008[0])
        self.state_control.unset_minion_status(self.app_id)

        self.self_terminate = False
        log.info('Minion finished')

    def process(self):
        log.info(_(
            'Spark minion (workflow_id=%s,app_id=%s) started (pid=%s)'),
            self.workflow_id, self.app_id, os.getpid())
        self.execute_process = multiprocessing.Process(
            name="minion", target=self.execute,
            args=(self.terminate_proc_queue,))
        self.execute_process.daemon = False

        self.ping_process = multiprocessing.Process(
            name="ping process", target=self.ping,
            args=(self.terminate_proc_queue,))
        self.ping_process.daemon = False

        self.execute_process.start()
        self.ping_process.start()

        # We join the following processes because this script only terminates by
        # explicitly receiving a SIGKILL signal.
        self.execute_process.join()
        self.ping_process.join()
        # https://stackoverflow.com/questions/29703063/python-multhttps://stackoverflow.com/questions/29703063/python-multiprocessing-queue-is-emptyiprocessing-queue-is-empty
        self.terminate_proc_queue.close()
        self.terminate_proc_queue.join_thread()
        sys.exit(0)

        # TODO: clean state files in the temporary directory after joining
        # (maybe pack it and persist somewhere else ?)
