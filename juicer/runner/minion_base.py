import json
import logging.config
import time

import datetime

import os
from juicer.runner.control import StateControlRedis

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.spark.spark_minion')


class Minion:
    MSG_PROCESSED = 'message_processed'

    def __init__(self, redis_conn, workflow_id, app_id, config):
        self.redis_conn = redis_conn
        self.state_control = StateControlRedis(self.redis_conn)
        self.workflow_id = workflow_id
        self.app_id = app_id
        self.config = config

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
        self.MNN010 = ('MNN010', _(
            'Task completed, but not executed (not used in the workflow).'))

        # Used in the template file, declared here to gettext detect them
        self.msgs = [
            _('Task running'), _('Task completed'),
            _('Task running (cached data)')
        ]

    def process(self):
        raise NotImplementedError()

    def _generate_output(self, message, status=None, code=None):
        """
        Sends feedback about execution of this minion.
        """
        obj = {'message': message, 'workflow_id': self.workflow_id,
               'app_id': self.app_id, 'code': code,
               'date': datetime.datetime.now().isoformat(),
               'status': status if status is not None else 'OK'}

        m = json.dumps(obj)
        self.state_control.push_app_output_queue(self.app_id, m)

    def _perform_ping(self):
        status = {
            'status': 'READY', 'pid': os.getpid(),
        }
        self.state_control.set_minion_status(
            self.app_id, json.dumps(status), ex=10, nx=False)

    def ping(self, q):
        """ Pings redis to inform master this minion is online """
        log.info('Start ping')
        while q.empty():
            self._perform_ping()
            time.sleep(5)
