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
import datetime

import codecs
import os
import socketio
from timeit import default_timer as timer
# noinspection PyCompatibility
from concurrent.futures import ThreadPoolExecutor
# noinspection PyCompatibility
from concurrent.futures import TimeoutError
from juicer.runner import configuration
from juicer.runner import juicer_protocol

from juicer.runner.minion_base import Minion
from juicer.spark.transpiler import SparkTranspiler
from juicer.util import dataframe_util, listener_util
from juicer.workflow.workflow import Workflow
from juicer.util.template_util import strip_accents

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.spark.spark_minion')

locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')


class SparkMinion(Minion):
    """
    Controls the execution of Spark code in Lemonade Juicer.
    """

    # max idle time allowed in seconds until this minion self termination
    IDLENESS_TIMEOUT = 600
    TIMEOUT = 'timeout'

    def __init__(self, redis_conn, workflow_id, app_id, config, lang='en',
                 jars=None):
        Minion.__init__(self, redis_conn, workflow_id, app_id, config)

        self.jars = jars
        self.terminate_proc_queue = multiprocessing.Queue()
        self.execute_process = None
        self.ping_process = None
        self.module = None

        self._state = {}
        self.transpiler = SparkTranspiler(config)
        self.config = config
        configuration.set_config(self.config)
        self.juicer_listener_enabled = False

        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')
        sys.path.append(self.tmp_dir)

        # Add pyspark to path
        spark_home = os.environ.get('SPARK_HOME')
        if spark_home:
            sys.path.append(os.path.join(spark_home, 'python'))
            log.info(_('SPARK_HOME set to %s'), spark_home)
        else:
            log.warn(_('SPARK_HOME environment variable is not defined'))

        self.spark_session = None
        signal.signal(signal.SIGTERM, self._terminate)

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
        #self._build_dist_file()

    def _build_dist_file(self):
        """
        Build a Zip file containing files in dist packages. Such packages
        contain code to be executed in the Spark cluster and should be
        distributed among all nodes.
        """
        project_base = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..', '..'))

        lib_paths = [
            project_base,
            os.path.join(project_base, 'juicer'),
            os.path.join(project_base, 'juicer', 'include'),
            os.path.join(project_base, 'juicer', 'privaaas'),
            os.path.join(project_base, 'juicer', 'runner'),
            os.path.join(project_base, 'juicer', 'service'),
            os.path.join(project_base, 'juicer', 'spark'),
            os.path.join(project_base, 'juicer', 'util'),
            os.path.join(project_base, 'juicer', 'workflow'),
            os.path.join(project_base, 'juicer', 'i18n/locales/pt/LC_MESSAGES'),
            os.path.join(project_base, 'juicer', 'i18n/locales/en/LC_MESSAGES'),
            os.path.join(project_base, 'juicer', 'i18n/locales/es/LC_MESSAGES'),
        ]
        valid_extensions = ['*.py', '*.ini', '*.mo']
        build = not os.path.exists(self.DIST_ZIP_FILE)

        def multiple_file_types(base_path, *patterns):
            return list(itertools.chain.from_iterable(
                glob.iglob(os.path.join(base_path, pattern)) for pattern in
                patterns))

        if not build:
            for lib_path in lib_paths:
                dist_files = multiple_file_types(lib_path, *valid_extensions)
                zip_mtime = os.path.getmtime(self.DIST_ZIP_FILE)
                for f in dist_files:
                    if zip_mtime < os.path.getmtime(
                            os.path.join(lib_path, f)):
                        build = True
                        break
                if build:
                    break

        if build:
            zf = zipfile.ZipFile(self.DIST_ZIP_FILE, mode='w')
            zf.pwd = project_base
            for lib_path in lib_paths:
                dist_files = multiple_file_types(lib_path, *valid_extensions)
                for f in dist_files:
                    zf.write(f, arcname=os.path.relpath(f, project_base))
            zf.close()

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
                              ee.message, '>\n'.join(tb))

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
            t.install(unicode=True)

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
            self.get_or_create_spark_session(loader, app_configs)

            # Mark job as running
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
                    loader.workflow, loader.graph, {}, out, job_id)

            # Get rid of .pyc file if it exists
            if os.path.isfile('{}c'.format(generated_code_path)):
                os.remove('{}c'.format(generated_code_path))

            self.module = importlib.import_module(module_name)
            self.module = imp.reload(self.module)
            if log.isEnabledFor(logging.debug):
                log.debug('Objects in memory after loading module: %s',
                          len(gc.get_objects()))

            # Starting execution. At this point, the transpiler have created a
            # module with a main function that receives a spark_session and the
            # current state (if any). We pass the current state to the execution
            # to avoid re-computing the same tasks over and over again, in case
            # of several partial workflow executions.
            new_state = self.module.main(
                self.get_or_create_spark_session(loader, app_configs),
                self._state,
                self._emit_event(room=job_id, namespace='/stand'))

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
            msg = ve.message
            txt = msg.decode('utf8') if isinstance(msg, str) else msg
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
                name='update job', message='\n'.join(tb),
                status='ERROR', identifier=job_id)
            self._generate_output(ee.message, 'ERROR', code=1000)
            result = False

        self.message_processed('execute')

        return result

    # noinspection PyProtectedMember
    def is_spark_session_available(self):
        """
        Check whether the spark session is available, i.e., the spark session
        is set and not stopped.
        """
        return (self.spark_session and
                self.spark_session.sparkContext._jsc and
                not self.spark_session.sparkContext._jsc.sc().isStopped())

    # noinspection PyUnresolvedReferences
    def get_or_create_spark_session(self, loader, app_configs):
        """
        Get an existing spark session (context) for this minion or create a new
        one. Ideally the spark session instantiation is done only once, in order
        to support partial workflow executions within the same context.
        """

        from pyspark.sql import SparkSession
        if not self.is_spark_session_available():
            log.info(_("Creating a new Spark session"))

            app_name = '%s(workflow_id=%s,app_id=%s)' % (
                strip_accents(loader.workflow.get('name', '')),
                self.workflow_id, self.app_id)

            spark_builder = SparkSession.builder.appName(
                app_name)

            # Use config file default configurations to set up Spark session
            for option, value in self.config['juicer'].get('spark', {}).items():
                if value is not None:
                    log.info(_('Setting spark configuration %s'), option)
                    spark_builder = spark_builder.config(option, value)

            # Set hadoop native libs, if available
            if "HADOOP_HOME" in os.environ:
                app_configs['driver-library-path'] = \
                    '{}/lib/native/'.format(os.environ.get('HADOOP_HOME'))

            # Default options from configuration file
            app_configs.update(self.config['juicer'].get('spark', {}))

            # Juicer listeners configuration.
            listeners = self.config['juicer'].get('listeners', [])

            classes = []
            all_jars = []
            for listener in listeners:
                clazz = listener['class']
                jars = listener['jars']
                classes.append(clazz)
                all_jars.extend(jars)
                if clazz == 'lemonade.juicer.spark.LemonadeSparkListener':
                    self.juicer_listener_enabled = True
                    app_configs['lemonade.juicer.eventLog.dir'] = \
                        listener.get('params', {}).get('log_path',
                                                       '/tmp/juicer-spark-logs')

            if self.jars:
                all_jars.extend(self.jars.split(os.path.pathsep))

            app_configs['spark.extraListeners'] = ','.join(classes)

            # Must use CLASSPATH from config file also!
            if 'spark.driver.extraClassPath' in app_configs:
                all_jars.append(app_configs['spark.driver.extraClassPath'])

            app_configs['spark.driver.extraClassPath'] = os.path.pathsep.join(
                all_jars)

            log.info('JAVA CLASSPATH: %s',
                     app_configs['spark.driver.extraClassPath'])
            # All options passed by application are sent to Spark
            for option, value in app_configs.items():
                spark_builder = spark_builder.config(option, value)

            self.spark_session = spark_builder.getOrCreate()
            # noinspection PyBroadException
            try:
                log_level = logging.getLevelName(log.getEffectiveLevel())
                self.spark_session.sparkContext.setLogLevel(log_level)
            except Exception as e:
                log_level = 'WARN'
                self.spark_session.sparkContext.setLogLevel(log_level)

                # self.transpiler.build_dist_file()
                # self.spark_session.sparkContext.addPyFile(
                #    self.transpiler.DIST_ZIP_FILE)

        log.info(_("Minion is using '%s' as Spark master"),
                 self.spark_session.sparkContext.master)
        return self.spark_session

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

    def terminate(self):
        """
        This is a handler that reacts to a sigkill signal. The most feasible
        scenario is when the JuicerServer is demanding the termination of this
        minion. In this case, we stop and release any allocated resource
        (spark_session) and kill the subprocess managed in here.
        """
        if self.spark_session:
            self.spark_session.stop()
            self.spark_session.sparkContext.stop()
            self.spark_session = None

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
