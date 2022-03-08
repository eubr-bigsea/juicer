# coding=utf-8
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

from io import StringIO
import pandas
import socketio
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
from juicer.runner import configuration
from juicer.runner import protocol as juicer_protocol

from juicer.runner.minion_base import Minion
from juicer.meta.transpiler import MetaTranspiler
# from juicer.spark.transpiler import SparkTranspiler
# from juicer.scikit_learn.transpiler import ScikitLearnTranspiler
from juicer.util import dataframe_util
from juicer.workflow.workflow import Workflow
from juicer.scikit_learn.scikit_learn_minion import ScikitLearnMinion
from juicer.spark.spark_minion import SparkMinion

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.meta.meta_minion')

locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')


class MetaMinion(Minion):
    """
    Controls the execution of Meta code in Lemonade Juicer.
    """

    # max idle time allowed in seconds until this minion self termination
    IDLENESS_TIMEOUT = 6000
    TIMEOUT = 'timeout'
    MSG_PROCESSED = 'message_processed'

    def __init__(self, redis_conn, workflow_id, app_id, config, lang='en'):
        """Initialize the minion."""
        Minion.__init__(self, redis_conn, workflow_id, app_id, config)

        self.terminate_proc_queue = multiprocessing.Queue()
        self.execute_process = None
        self.ping_process = None
        self.module = None

        self.config = config

        self.transpiler = MetaTranspiler(config)
        configuration.set_config(self.config)

        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')
        sys.path.append(self.tmp_dir)

        signal.signal(signal.SIGTERM, self._terminate)

        self.mgr = socketio.RedisManager(
            config['juicer']['servers']['redis_url'],
            'job_output')

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job_future = None

        self.meta_config = config['juicer'].get('meta', {})

        # self termination timeout
        self.active_messages = 0
        self.self_terminate = True
        self.current_lang = lang
        self.target_minion = None

    def _emit_event(self, room, namespace):
        def emit_event(name, message, status, identifier, **kwargs):
            log.debug(_('Emit %s %s %s %s'), name, message, status,
                      identifier)
            data = {'message': message, 'status': status, 'id': identifier}
            data.update(kwargs)
            # print('+' * 20)
            # print(name, data, room, namespace)
            # print('+' * 20)
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
        if msg is None:
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

            self.current_lang = msg_info.get('app_configs', {}).get(
                'locale', self.current_lang)
            lang = self.current_lang
            t = gettext.translation('messages', locales_path, [lang],
                                    fallback=True)
            t.install()

            self._emit_event(room=job_id, namespace='/stand')(
                name='update job',
                message=_('Running job with lang {}/{}').format(
                    lang, self.current_lang),
                status='RUNNING', identifier=job_id)

            app_configs = msg_info.get('app_configs', {})

            # Sample size can be informed in API, limited to 1000 rows.

            self.transpiler.sample_size = min(1000, int(app_configs.get(
                'sample_size', 50)))

            self.job_future = self._execute_future(job_id, workflow,
                                                   app_configs)
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

    def _execute_future(self, job_id, workflow, app_configs):
        #return self.executor.submit(self.perform_execute,
        #                            job_id, workflow, app_configs)
        return self.perform_execute(job_id, workflow, app_configs)

    def _get_target_workflow(self, job_id, workflow, app_configs, target_platform,
            include_disabled=False):
        loader = Workflow(workflow, self.config, lang=self.current_lang,
            include_disabled=include_disabled)
        loader.handle_variables({'job_id': job_id})
        out = StringIO()
        # print('-' * 20)
        # print(loader.workflow, loader.graph.nodes())
        #loader.workflow['disabled_tasks'] = []
        # print('-' * 20)
        self.transpiler.target_platform = target_platform
        self.transpiler.transpile(loader.workflow, loader.graph,
            self.config, out, job_id, persist=app_configs.get('persist'))
        out.seek(0)
        target_workflow = json.loads(out.read())
        target_workflow['app_configs'] = app_configs
        return target_workflow

    def perform_execute(self, job_id, workflow, app_configs):
        if workflow.get('type') == 'MODEL_BUILDER':
            self._execute_model_builder(job_id, workflow, app_configs)
        else:
            self._execute_target_workflow(job_id, workflow, app_configs)

    def _execute_model_builder(self, job_id, workflow, app_configs):
        loader = Workflow(workflow, self.config, lang=self.current_lang)
        loader.handle_variables({'job_id': job_id})
        out = StringIO()

        self.transpiler.transpile(loader.workflow, loader.graph,
            self.config, out, job_id, persist=app_configs.get('persist'))
        out.seek(0)
        code = out.read()

        if self.target_minion is None:
            # Only Spark is supported
            self.target_minion = SparkMinion(
                self.redis_conn, self.workflow_id,
                self.app_id, self.config, self.current_lang)

        self.target_minion.perform_execute(job_id, workflow, app_configs, code)

    def _execute_target_workflow(self, job_id, workflow, app_configs):
        app_configs['persist'] = False

        log.info('Converting workflow to platform %s',
            app_configs.get('target_platform', 'scikit_learn'))
        target_workflow = self._get_target_workflow(job_id, workflow,
            app_configs, None)

        # REMOVE: Auto plug allows to connect ports automatically if they're compatible
        # app_configs['auto_plug'] = True

        if self.target_minion is None:
            if app_configs.get('target_platform') == 'spark':
                self.target_minion = SparkMinion(
                    self.redis_conn, self.workflow_id,
                    self.app_id, self.config, self.current_lang)
            else:
                self.target_minion = ScikitLearnMinion(
                    self.redis_conn, self.workflow_id,
                    self.app_id, self.config, self.current_lang)

        #return self.executor.submit(
        #    self.target_minion.perform_execute,
        #    job_id, target_workflow, app_configs)

        # print('*' * 20)
        # print(self.target_minion._state)
        # print('*' * 20)
        return self.target_minion.perform_execute(
            job_id, target_workflow, app_configs)


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

    def message_processed(self, msg_type, wid, job_id, workflow):
        msg_processed = {
            'workflow_id': wid,
            'app_id': wid,
            'job_id': job_id,
            'workflow': workflow,
            'type': MetaMinion.MSG_PROCESSED,
            'msg_type': msg_type,

        }
        self.state_control.push_app_queue(self.app_id,
                                          json.dumps(msg_processed))

    # noinspection PyUnusedLocal
    def _terminate(self, _signal, _frame):
        self.terminate()

    def terminate(self):
        """
        This is a handler that reacts to a sigkill signal. The most feasible
        scenario is when the JuicerServer is demanding the termination of this
        minion. In this case, we stop and release any allocated resource
        and kill the subprocess managed in here.
        """
        log.info('Post terminate message in queue')
        self.terminate_proc_queue.put({'terminate': True})

        message = self.MNN008[1].format(self.app_id)
        log.info(message)
        self._generate_output(message, 'SUCCESS', self.MNN008[0])
        self.state_control.unset_minion_status(self.app_id)

        self.self_terminate = False
        log.info('Minion finished, pid = %s (%s)',
                os.getpid(), multiprocessing.current_process().name)
        self.state_control.shutdown()
        sys.exit(0)

    def process(self):
        log.info(_(
            'Meta minion (workflow_id=%s,app_id=%s) started (pid=%s)'),
            self.workflow_id, self.app_id, os.getpid())
        self.execute_process = multiprocessing.Process(
            name="minion", target=self.execute,
            args=(self.terminate_proc_queue,))
        self.execute_process.daemon = True

        self.ping_process = multiprocessing.Process(
            name="ping process", target=self.ping,
            args=(self.terminate_proc_queue,))
        self.ping_process.daemon = True

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
