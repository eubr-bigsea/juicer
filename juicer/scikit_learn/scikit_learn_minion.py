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
import yaml
import socketio
from kubernetes import client
from kubernetes.client.rest import ApiException

from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
from juicer.runner import configuration
from juicer.runner import protocol as juicer_protocol

from juicer.runner.minion_base import Minion
from juicer.scikit_learn.transpiler import ScikitLearnTranspiler
from juicer.util import dataframe_util
from juicer.workflow.workflow import Workflow

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.scikit_learn.scikit_learn_minion')

locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')


class ScikitLearnMinion(Minion):
    """
    Controls the execution of Scikit-Learn code in Lemonade Juicer.
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

        self.transpiler = ScikitLearnTranspiler(config)
        configuration.set_config(self.config)

        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')
        sys.path.append(self.tmp_dir)

        signal.signal(signal.SIGTERM, self._terminate)

        self.mgr = socketio.RedisManager(
            config['juicer']['servers']['redis_url'],
            'job_output')

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job_future = None

        self.scikit_learn_config = config['juicer'].get('scikit_learn', {})

        # self termination timeout
        self.active_messages = 0
        self.self_terminate = True
        self.juicer_listener_enabled = False
        self.current_lang = lang
        self.cluster_options = {}

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

    def create_configmap(self, client, namespace, generated_code_path,
                         configuration, name):

        api_instance = client.CoreV1Api(client.ApiClient(configuration))

        # Configureate ConfigMap metadata
        metadata = client.V1ObjectMeta(
                name=name,
                namespace=namespace
        )
        code = generated_code_path.replace('/tmp/', '')
        # Get File Content
        with open(generated_code_path, 'r') as f:
            file_content = f.read()
        # Instantiate the configmap object
        configmap = client.V1ConfigMap(
                api_version="v1",
                kind="ConfigMap",
                data={code: file_content},
                metadata=metadata,
        )

        try:
            api_instance.create_namespaced_config_map(
                    namespace=namespace,
                    body=configmap,
                    pretty=True
            )

        except ApiException as e:
            print("Exception when calling CoreV1Api->"
                  "create_namespaced_config_map: %s\n" % e)

    def create_k8s_job(self, job_id, cluster_options, generated_code_path):

        configuration = client.Configuration()
        configuration.host = cluster_options['address']
        configuration.verify_ssl = False
        configuration.debug = False

        env_vars = {
            'HADOOP_CONF_DIR': '/usr/local/juicer/conf',
        }

        token = cluster_options['auth_token']
        configuration.api_key = {"authorization": "Bearer " + token}
        # noinspection PyUnresolvedReferences
        client.Configuration.set_default(configuration)

        job = client.V1Job(api_version="batch/v1", kind="Job")
        name = 'job-{}-{}-{}'.format(job_id, self.workflow_id, self.app_id)
        code_configmap = name
        container_name = 'juicer-job'
        container_image = cluster_options['container']
        namespace = cluster_options['namespace']
        minion_cmd = ["python", generated_code_path]

        job.metadata = client.V1ObjectMeta(namespace=namespace, name=name)
        job.status = client.V1JobStatus()

        # Now we start with the Template...
        template = client.V1PodTemplate()
        template.template = client.V1PodTemplateSpec()

        # Passing Arguments in Env:
        env_list = []
        for env_name, env_value in env_vars.items():
            env_list.append(client.V1EnvVar(name=env_name, value=env_value))

        self.create_configmap(client, namespace,
                              generated_code_path, configuration,
                              code_configmap)

        # Subpath implies that the file is stored as a config map in kb8s
        volume_mounts = [
            client.V1VolumeMount(
                    name='juicer-config', sub_path='juicer-config.yaml',
                    mount_path='/usr/local/juicer/conf/juicer-config.yaml'),
            client.V1VolumeMount(
                    name='hdfs-site', sub_path='hdfs-site.xml',
                    mount_path='/usr/local/juicer/conf/hdfs-site.xml'),
            client.V1VolumeMount(
                    name='hdfs-pvc',
                    mount_path='/srv/storage/'),
            client.V1VolumeMount(
                    name=code_configmap,
                    sub_path=generated_code_path.replace('/tmp/', ''),
                    mount_path=generated_code_path),
        ]
        pvc_claim = client.V1PersistentVolumeClaimVolumeSource(
                claim_name='hdfs-pvc')

        # resources = {'limits': {'nvidia.com/gpu': 1}}
        resources = {}

        container = client.V1Container(name=container_name,
                                       image=container_image,
                                       env=env_list, command=minion_cmd,
                                       image_pull_policy='Always',
                                       volume_mounts=volume_mounts,
                                       resources=resources)

        volumes = [
            client.V1Volume(
                    name='juicer-config',
                    config_map=client.V1ConfigMapVolumeSource(
                        name='juicer-config')),
            client.V1Volume(
                    name='hdfs-site',
                    config_map=client.V1ConfigMapVolumeSource(
                        name='hdfs-site')),
            client.V1Volume(
                    name=code_configmap,
                    config_map=client.V1ConfigMapVolumeSource(
                            name=code_configmap)),
            client.V1Volume(name='hdfs-pvc',
                            persistent_volume_claim=pvc_claim),
        ]
        template.template.spec = client.V1PodSpec(
                containers=[container], restart_policy='Never', volumes=volumes)

        # And finally we can create our V1JobSpec!
        job.spec = client.V1JobSpec(ttl_seconds_after_finished=10,
                                    template=template.template)
        api = client.ApiClient(configuration)
        batch_api = client.BatchV1Api(api)

        try:
            batch_api.create_namespaced_job(namespace, job, pretty=True)
        except ApiException as e:
            print("Exception when calling BatchV1Api->: %s\n" % e)

        # check if job is completed
        running = True
        try:
            while running:
                status = batch_api.read_namespaced_job_status(
                        name=name, namespace=namespace, pretty=True).status

                running = status.active == 1
                failed = status.failed == 1
        except ApiException as e:
            print("Exception when calling BatchV1Api->: %s\n" % e)

        # remove configmap
        api_instance = client.CoreV1Api(client.ApiClient(configuration))
        body = client.V1DeleteOptions()
        try:
            api_instance.delete_namespaced_config_map(name, namespace,
                                                      body=body)

        except ApiException as e:
            print("Exception when calling CoreV1Api->"
                  "delete_namespaced_config_map: %s\n" % e)

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
            cluster_info = msg_info.get('cluster', {})
            workflow = msg_info['workflow']
            lang = workflow.get('locale', self.current_lang)

            cluster_type = cluster_info.get('type', 'SPARK_LOCAL')
            if cluster_type not in ('SPARK_LOCAL', 'KUBERNETES'):
                self._emit_event(room=job_id, namespace='/stand')(
                        name='update job',
                        message=_('Unsupported cluster type, '
                                  'it cannot run Spark applications.'),
                        status='ERROR', identifier=job_id)
                return

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

            self.cluster_options['address'] = cluster_info['address']
            self.cluster_options['auth_token'] = cluster_info['auth_token']

            log.info("Cluster options: %s",
                     json.dumps(self.cluster_options, indent=0))

            self._emit_event(room=job_id, namespace='/stand')(
                name='update job',
                message=_('Running job with lang {}/{}').format(
                    lang, self.current_lang),
                status='RUNNING', identifier=job_id)

            t = gettext.translation('messages', locales_path, [lang],
                                    fallback=True)
            t.install()

            app_configs = msg_info.get('app_configs', {})

            if self.job_future:
                self.job_future.result()

            self.job_future = self._execute_future(job_id, workflow,
                                                   app_configs, cluster_type,
                                                   self.cluster_options)
            log.info(_('Execute message finished'))
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

    def _execute_future(self, job_id, workflow, app_configs, cluster_type,
                        cluster_options):

        if cluster_type == 'KUBERNETES':

            return self._perform_execute_k8s(job_id, workflow, app_configs,
                                             cluster_options)

        else:
            return self.executor.submit(self._perform_execute_local,
                                        job_id, workflow, app_configs)

    def _perform_execute_k8s(self, job_id, workflow, app_configs, cluster_options):

        # Sleeps 1s in order to wait for client join notification room
        time.sleep(1)
        result = True
        start = timer()
        try:
            loader = Workflow(workflow, self.config)

            # force the scikit-learn context creation
            self.get_or_create_scikit_learn_session(loader, app_configs, job_id)

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

            with codecs.open(generated_code_path, 'w', 'utf8') as out:
                self.transpiler.transpile(
                    loader.workflow, loader.graph, {}, out, job_id)

            # Get rid of .pyc file if it exists
            if os.path.isfile('{}c'.format(generated_code_path)):
                os.remove('{}c'.format(generated_code_path))

            # Launch the scikit_learn
            self.create_k8s_job(job_id, cluster_options, generated_code_path)

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
            print(('#' * 30))
            import traceback
            traceback.print_exc(file=sys.stdout)
            print(('#' * 30))
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
                message=_('Internal error.'),
                name='update job', exception_stack='\n'.join(tb),
                status='ERROR', identifier=job_id)
            self._generate_output(str(ee), 'ERROR', code=1000)
            result = False

        self.message_processed('execute')

        return result

    def _perform_execute_local(self, job_id, workflow, app_configs):

        # Sleeps 1s in order to wait for client join notification room
        time.sleep(1)

        result = True
        start = timer()
        try:
            loader = Workflow(workflow, self.config)

            # force the scikit-learn context creation
            self.get_or_create_scikit_learn_session(loader, app_configs, job_id)

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

            with codecs.open(generated_code_path, 'w', 'utf8') as out:
                self.transpiler.transpile(
                    loader.workflow, loader.graph, {}, out, job_id)

            # Get rid of .pyc file if it exists
            if os.path.isfile('{}c'.format(generated_code_path)):
                os.remove('{}c'.format(generated_code_path))

            # Launch the scikit_learn
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
                    self.get_or_create_scikit_learn_session(loader,
                                                            app_configs,
                                                            job_id),
                    self._state,
                    self._emit_event(room=job_id, namespace='/stand'))
            except:
                raise

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
            print(('#' * 30))
            import traceback
            traceback.print_exc(file=sys.stdout)
            print(('#' * 30))
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
                message=_('Internal error.'),
                name='update job', exception_stack='\n'.join(tb),
                status='ERROR', identifier=job_id)
            self._generate_output(str(ee), 'ERROR', code=1000)
            result = False

        self.message_processed('execute')

        return result

    # noinspection PyUnresolvedReferences,PyMethodMayBeStatic
    def get_or_create_scikit_learn_session(self, loader, app_configs, job_id):
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
            'type': ScikitLearnMinion.MSG_PROCESSED,
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
