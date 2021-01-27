# coding=utf-8


import importlib
import json
import logging.config
import time

import datetime

import os
import pyinotify
from juicer.runner.control import StateControlRedis

# noinspection PyUnresolvedReferences
from six.moves import reload_module

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.spark.spark_minion')

_watch_dir = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, os.pardir))


# noinspection PyPep8Naming,PyMethodMayBeStatic
class EventHandler(pyinotify.ProcessEvent):
    allowed_extensions = ["py"]

    def is_allowed_path(self, filename, is_dir):
        # Don't check the extension for directories
        if not is_dir:
            ext = os.path.splitext(filename)[1][1:].lower()
            if ext not in self.allowed_extensions:
                return False
        return True

    def _reload(self, event):
        if self.is_allowed_path(event.pathname, event.dir):
            module = importlib.import_module(
                event.pathname.replace(_watch_dir, '')[1:-3].replace('/', '.'))
            reload_module(module)
            log.warn(_('Reloading {}'.format(module)))

    def process_IN_MODIFY(self, event):
        self._reload(event)

    def process_IN_MOVED_TO(self, event):

        self._reload(event)


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

        self.MNN011 = ('MNN011', _(
            'Error accessing data. Probably attribute "{}" does not exist.'))
        # Used in the template file, declared here to gettext detect them
        self.msgs = [
            _('Task running'), _('Task completed'),
            _('Task running (cached data)')
        ]
        self.pid = os.getpid()

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
            'status': 'READY', 'pid': self.pid,
        }
        self.state_control.set_minion_status(
            self.app_id, json.dumps(status), ex=10, nx=False)

    @staticmethod
    def reload_code(q):
        wm = pyinotify.WatchManager()
        notifier = pyinotify.Notifier(wm, EventHandler())
        wm.add_watch(_watch_dir, pyinotify.ALL_EVENTS, rec=True)
        notifier.loop()

    def ping(self, q):
        """ Pings redis to inform master this minion is online """
        log.info('Start ping')
        while q.empty():
            self._perform_ping()
            time.sleep(5)
