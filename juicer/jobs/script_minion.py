# coding=utf-8

import gc
import gettext
import imp
import importlib
import json
import logging.config
import multiprocessing
import signal
import sys
import time
import traceback

# noinspection PyUnresolvedReferences
import zipfile
import glob

import itertools

import codecs
import os
import socketio
from timeit import default_timer as timer
# noinspection PyCompatibility
from concurrent.futures import ThreadPoolExecutor
# noinspection PyCompatibility
from concurrent.futures import TimeoutError
from juicer.runner import configuration
from juicer.runner import protocol as juicer_protocol

from juicer.runner.minion_base import Minion
from juicer.util import dataframe_util, listener_util
from juicer.workflow.workflow import Workflow
from juicer.util.template_util import strip_accents

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.script.script_minion')

locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')


class ScriptMinion(Minion):
    """
    Controls the execution of Spark code in Lemonade Juicer.
    """

    # max idle time allowed in seconds until this minion self termination
    IDLENESS_TIMEOUT = 600
    TIMEOUT = 'timeout'

    def __init__(self, redis_conn, workflow_id, app_id, config, lang='en'):
        Minion.__init__(self, redis_conn, workflow_id, app_id, config)

        self.terminate_proc_queue = multiprocessing.Queue()
        self.execute_process = None
        self.ping_process = None
        self.reload_code_process = None
        self.module = None

        self._state = {}
        self.transpiler = SparkTranspiler(config)
        self.config = config
        configuration.set_config(self.config)
        self.juicer_listener_enabled = False

        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')
        sys.path.append(self.tmp_dir)

        self.mgr = socketio.RedisManager(
            config['juicer']['servers']['redis_url'],
            'job_output')

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job_future = None

        # self termination timeout
        self.active_messages = 0
        self.self_terminate = True
        # Errors and messages
        self.MNN000 = ('MNN000', _('Success.'))
        self.MNN001 = ('MNN001', _('Port output format not supported.'))
        self.MNN002 = ('MNN002', _('Success getting data from task.'))
        self.MNN003 = ('MNN003', _('State does not exists, processing app.'))
        self.MNN004 = ('MNN004', _('Invalid port.'))
        self.MNN005 = ('MNN005',
                       _('Unable to retrieve data because a previous error.'))
        self.MNN006 = ('MNN006',
                       _('Invalid Python code or incorrect encoding: {}'))
        self.MNN007 = ('MNN007', _('Job {} was canceled'))
        self.MNN008 = ('MNN008', _('App {} was terminated'))
        self.MNN009 = ('MNN009', _('Workflow specification is missing'))

        # Used in the template file, declared here to gettext detect them
        self.msgs = [
            _('Task running'), _('Task completed'),
            _('Task ignored (not used by other task or as an output)')
        ]
        self.current_lang = lang
        # self._build_dist_file()
        signal.signal(signal.SIGTERM, self._terminate)
        signal.signal(signal.SIGINT, self._cleanup)
        self.last_job_id = 0
        self.new_session = False

        self.cluster_options = {}
        self.last_cluster_id = None
        self.DIST_ZIP_FILE = '/tmp/lemonade-lib-pythoni_{}.zip'.format(
            self.app_id)

    def _cleanup(self, pid, flag):
        log.warn(_('Finishing minion'))
        msg = _('Pressed CTRL+C / SIGINT. Minion canceled the job.')
        self._emit_event(room=self.last_job_id, namespace='/stand')(
            name='update job', message=msg,
            status='ERROR', identifier=self.last_job_id)
        self.terminate()
        sys.exit(0)


    def _emit_event(self, room, namespace):
        def emit_event(name, message, status, identifier, **kwargs):
            log.debug(_('Emit %s %s %s %s'), name, message, status,
                      identifier)
            data = {'message': message, 'status': status, 'id': identifier}
            data.update(kwargs)
            self.mgr.emit(name, data=data, room=str(room), namespace=namespace)

            if not self.juicer_listener_enabled:
                return

            if name == 'update task':
                if status == 'RUNNING':
                    listener_util.post_event_to_spark(self.spark_session,
                                                      listener_util.TASK_START,
                                                      data)

                elif status == 'COMPLETED' or status == 'ERROR':
                    listener_util.post_event_to_spark(self.spark_session,
                                                      listener_util.TASK_END,
                                                      data)

            elif name == 'update job':
                if status == 'RUNNING':
                    listener_util.post_event_to_spark(self.spark_session,
                                                      listener_util.JOB_START,
                                                      data)

                elif status == 'COMPLETED' or status == 'ERROR':
                    listener_util.post_event_to_spark(self.spark_session,
                                                      listener_util.JOB_END,
                                                      data)

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

        if msg is None:
            if self.active_messages == 0:
                self._timeout_termination()
            return

        try:
            msg_info = json.loads(msg)
        except TypeError as te:
            log.exception(_('Invalid message JSON: {}').format(msg))
            return

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

            # Checks if it's a valid cluster
            job_id = msg_info['job_id']
            cluster_info = msg_info.get('cluster', {})
            cluster_type = cluster_info.get('type', 'SPARK_LOCAL')
            if cluster_type not in ('SPARK_LOCAL', 'MESOS', 'YARN',
                                    'KUBERNETES'):
                self._emit_event(room=job_id, namespace='/stand')(
                    name='update job',
                    message=_('Unsupported cluster type, '
                              'it cannot run Spark applications.'),
                    status='ERROR', identifier=job_id)
                return

            if all([self.last_cluster_id,
                    self.last_cluster_id != cluster_info['id']]):
                if self.spark_session:
                    self._emit_event(room=job_id, namespace='/stand')(
                        name='update job',
                        message=_('Cluster configuration changed. '
                                  'Stopping previous cluster.'),
                        status='RUNNING', identifier=job_id)
                    # Requires finish Spark Context
                    self.spark_session.stop()
                    self._state = {}
                    self.spark_session = None

            self.cluster_options = {}

            # Add general parameters in the form param1=value1,param2=value2
            try:
                if cluster_info.get('general_parameters'):
                    parameters = cluster_info['general_parameters'].split(',')
                    for parameter in parameters:
                        key, value = parameter.split('=')
                        self.cluster_options[key.strip()] = value.strip()
            except Exception as ex:
                msg = _("Error in general cluster parameters: {}").format(ex)
                self._emit_event(room=job_id, namespace='/stand')(
                    name='update job',
                    message=msg,
                    status='CANCELED', identifier=job_id)
                log.warn(msg)
                return

                # Spark mapping for cluster properties
            options = {'address': 'spark.master',
                       'executors': 'spark.cores.max',
                       'executor_cores': 'spark.executor.cores',
                       'executor_memory': 'spark.executor.memory',
                       }

            if cluster_type == "KUBERNETES":
                options['executors'] = 'spark.executor.instances'

            for option, spark_name in list(options.items()):
                self.cluster_options[spark_name] = cluster_info[option]

            log.info("Cluster options: %s",
                     json.dumps(self.cluster_options, indent=0))

            self.last_cluster_id = cluster_info['id']

            self.active_messages += 1
            log.info('Execute message received')
            self.last_job_id = job_id
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
            self.active_messages += 1
            log.info('Deliver message received')
            task_id = msg_info.get('task_id')
            output = msg_info.get('output')
            port = msg_info.get('port')
            job_id = msg_info['job_id']
            workflow = msg_info.get('workflow')
            app_configs = msg_info.get('app_configs', {})

            if self.job_future:
                self.job_future.result()

            self.job_future = self._deliver_future(task_id,
                                                   output, port, job_id,
                                                   workflow, app_configs)

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

        elif msg_type == SparkMinion.MSG_PROCESSED:
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
            # force the spark context creation
            self.get_or_create_spark_session(loader, app_configs, job_id)

            # Mark job as running
            if self.new_session:
                self._emit_event(room=job_id, namespace='/stand')(
                    name='update job',
                    message=_('Running job, but it requires allocation'
                              ' of '
                              'cluster computers first and it takes time.'),
                    status='RUNNING', identifier=job_id)
            else:
                self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', message=_('Running job'),
                    status='RUNNING', identifier=job_id)

            module_name = 'juicer_app_{}_{}_{}'.format(
                self.workflow_id,
                self.app_id,
                job_id)

            generated_code_path = os.path.join(
                self.tmp_dir, module_name + '.py')

            with codecs.open(generated_code_path, 'w', 'utf8') as out:
                self.transpiler.transpile(
                    loader.workflow, loader.graph, {}, out, job_id,
                    self._state)

            # Get rid of .pyc file if it exists
            if os.path.isfile('{}c'.format(generated_code_path)):
                os.remove('{}c'.format(generated_code_path))

            self.module = importlib.import_module(module_name)
            self.module = imp.reload(self.module)
            if log.isEnabledFor(logging.DEBUG):
                log.debug('Objects in memory after loading module: %s',
                          len(gc.get_objects()))

            # Starting execution. At this point, the transpiler have created a
            # module with a main function that receives a spark_session and the
            # current state (if any). We pass the current state to the execution
            # to avoid re-computing the same tasks over and over again, in case
            # of several partial workflow executions.
            try:
                new_state = self.module.main(
                    self.get_or_create_spark_session(loader, app_configs,
                                                     job_id),
                    self._state,
                    self._emit_event(room=job_id, namespace='/stand'))
            except Exception as ex:
                if self.is_spark_session_available():
                    self.spark_session.sparkContext.cancelAllJobs()
                raise ex from None

            end = timer()
            # Mark job as completed
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job',
                message=_('Job finished in {0:.2f}s').format(end - start),
                status='COMPLETED', identifier=job_id)

            # We update the state incrementally, i.e., new task results can be
            # overwritten but never lost.
            self._state.update(new_state)

        except UnicodeEncodeError as ude:
            message = self.MNN006[1].format(ude)
            log.exception(_(message))
            # Mark job as failed
            self._emit_event(room=job_id, namespace='/stand')(
                name='update job', message=message,
                status='ERROR', identifier=job_id)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False

        except ValueError as ve:
            txt = str(ve)
            message = _('Invalid or missing parameters: {}').format(txt)
            log.exception(message)
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
            log.exception(message)
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
                exception_stack='\n'.join(tb),
                message=_('Unhandled error'),
                name='update job',
                status='ERROR', identifier=job_id)
            self._generate_output(ee, 'ERROR', code=1000)
            result = False

        self.message_processed('execute')

        stop = self.config['juicer'].get('minion', {}).get(
            'terminate_after_run', False)

        if stop:
            log.warn(
                _('Minion is configured to stop Spark after each execution'))
            self._state = {}
            self.spark_session.stop()
            self.spark_session = None

        return result


    def _send_to_output(self, data):
        self.state_control.push_app_output_queue(
            self.app_id, json.dumps(data))

    def _send_delivery(self, output, status_data, csv_rows=None):
        if csv_rows is None:
            csv_rows = []
        msg = {}
        msg.update(status_data)
        msg['sample'] = '\n'.join(csv_rows)
        self.state_control.push_queue(output, json.dumps(msg))

    def _deliver_future(self, task_id, output, port, job_id, workflow,
                        app_configs):
        return self.executor.submit(self._perform_deliver, task_id, output,
                                    port, job_id, workflow, app_configs)

    def _perform_deliver(self, task_id, output, port, job_id, workflow,
                         app_configs):

        data = []

        # FIXME: state must expire. How to identify this in the interface?
        # FIXME: Define how to identify the request.
        # FIXME: Define where to store generated data (Redis?)
        if task_id in self._state:
            success, status_data, data = \
                self._read_dataframe_data(task_id, output, port)
        elif workflow:
            self._send_to_output({
                'status': 'WARNING',
                'code': self.MNN003[0],
                'message': self.MNN003[1]
            })

            # FIXME: Report missing or process workflow until this task
            if self._perform_execute(job_id, workflow, app_configs):
                success, status_data, data = \
                    self._read_dataframe_data(task_id, output, port)
            else:
                status_data = {'status': 'ERROR', 'code': self.MNN005[0],
                               'message': self.MNN005[1]}
                success = False
        else:
            status_data = {'status': 'ERROR', 'code': self.MNN009[0],
                           'message': self.MNN009[1]}
            success = False

        self._send_to_output(status_data)
        self._send_delivery(output, status_data, data)

        self.message_processed('deliver')

        return success


    # noinspection PyUnusedLocal
    def cancel_job(self, job_id):
        if self.job_future:
            while True:
                if self.is_spark_session_available():
                    self.spark_session.sparkContext.cancelAllJobs()
                try:
                    self.job_future.result(timeout=1)
                    break
                except TimeoutError as te:
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
            'type': SparkMinion.MSG_PROCESSED,
            'msg_type': msg_type
        }
        self.state_control.push_app_queue(self.app_id,
                                          json.dumps(msg_processed))
        log.info('Sending message processed message: %s' % msg_processed)

    # noinspection PyUnusedLocal
    def _terminate(self, _signal, _frame):
        self.terminate()

    # noinspection PyProtectedMember
    def terminate(self):
        """
        This is a handler that reacts to a sigkill signal. The most feasible
        scenario is when the JuicerServer is demanding the termination of this
        minion. In this case, we stop and release any allocated resource
        (spark_session) and kill the subprocess managed in here.
        """
        if self.spark_session:
            sc = self.spark_session.sparkContext

            self.spark_session.stop()
            self.spark_session.sparkContext.stop()
            self.spark_session = None
            sc._gateway.shutdown_callback_server()

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

        self.reload_code_process = multiprocessing.Process(
            name="reload code process", target=self.reload_code,
            args=(self.terminate_proc_queue,))
        self.reload_code_process.daemon = False

        self.execute_process.start()
        self.ping_process.start()
        # self.reload_code_process.start()

        # We join the following processes because this script only terminates by
        # explicitly receiving a SIGKILL signal.
        self.execute_process.join()
        self.ping_process.join()
        # self.reload_code_process.join()
        # https://stackoverflow.com/questions/29703063/python-multiprocessing-queue-is-emptyiprocessing-queue-is-empty
        self.terminate_proc_queue.close()
        self.terminate_proc_queue.join_thread()
        sys.exit(0)

        # TODO: clean state files in the temporary directory after joining
        # (maybe pack it and persist somewhere else ?)
