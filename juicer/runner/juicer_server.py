# coding=utf-8
"""
"
"""
import argparse
import errno
import json
import logging
import multiprocessing
import signal
import subprocess
import sys
import time
import urlparse

import os
import redis
import yaml
from juicer.exceptions import JuicerException
from juicer.runner.control import StateControlRedis

logging.basicConfig(
    format=('[%(levelname)s] %(asctime)s,%(msecs)05.1f '
            '(%(funcName)s:%(lineno)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class JuicerServer:
    """
    The JuicerServer is responsible for managing the lifecycle of minions.
    A minion controls a application, i.e., an active instance of an workflow.
    Thus, the JuicerServer receives launch request from clients, launches and
    manages minion processes and takes care of their properly termination.
    """
    STARTED = 'STARTED'
    LOADED = 'LOADED'
    TERMINATION_PILL = 'terminate'
    HELP_UNHANDLED_EXCEPTION = 1
    HELP_STATE_LOST = 2

    def __init__(self, config, minion_executable, log_dir='/tmp',
                 config_file_path=None):

        self.minion_support_process = None
        self.start_process = None
        self.minion_status_process = None

        self.active_minions = {}

        self.config = config
        self.config_file_path = config_file_path
        self.minion_executable = minion_executable
        self.log_dir = log_dir or \
                self.config['juicer'].get('log', {}).get('path', '/tmp')

    def start(self):
        log.info('Starting master process. Reading "start" queue ')

        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
            port=parsed_url.port)

        while True:
            self.read_start_queue(redis_conn)

    # noinspection PyMethodMayBeStatic
    def read_start_queue(self, redis_conn):

        # Process next message
        state_control = StateControlRedis(redis_conn)
        msg = state_control.pop_start_queue()
        item = json.loads(msg)

        try:
            app_id = item.get('app_id')
            workflow_id = item.get('workflow_id')
            app_id = str(app_id) if app_id else None
            workflow_id = str(workflow_id) if workflow_id else None

            if app_id is None:
                raise ValueError('Application id not informed')
            elif item.get(self.TERMINATION_PILL, 'false').lower() in ('true'):
                self._terminate_minion(app_id)
                return

            # NOTE: Currently we are assuming that clients will only submit one
            # workflow at a time. Such assumption implies that both workflow_id
            # and app_id must be individually unique in this server at any point
            # of time. We plan to change that in the future, by allowing
            # multiple instances of the same workflow to be launched
            # concurrently.
            if self.active_minions.get(app_id, None) and \
                    state_control.get_workflow_status(workflow_id) != self.STARTED:
                raise JuicerException('Workflow {} should be started'.format(
                    workflow_id), code=1000)
            
            # Get minion status, if it exists
            minion_info = state_control.get_minion_status(app_id)
            log.debug('Minion status for app %s: %s', app_id, minion_info)

            # If there is status registered for the application then we do not
            # need to launch a minion for it, because it is already running.
            # Otherwise, we launch a new minion for the application.
            if minion_info:
                log.debug('Minion %s is running.', 'minion_{}'.format(app_id))
            else:
                minion_process = self._start_minion(
                        workflow_id, app_id, state_control)
                self.active_minions[app_id] = minion_process
            
            # Make the message (workflow partial execution) visible to the
            # minion, which will pop this set of commands and submit to the
            # underlying spark context
            state_control.push_app_queue(app_id, msg)
            state_control.set_workflow_status(workflow_id, self.STARTED)

            log.info('Generating code for app %s', app_id)
            state_control.push_app_output_queue(app_id, json.dumps(
                {'code': 0, 'message': 'Workflow will start soon'}))

        except JuicerException as je:
            log.error(je)
            if app_id:
                state_control.push_app_output_queue(app_id, json.dumps(
                    {'code': je.code, 'message': je.message}))
        except Exception as ex:
            log.error(ex)
            if app_id:
                state_control.push_app_output_queue(
                    app_id, json.dumps({'code': 500, 'message': ex.message}))

    def _start_minion(self, workflow_id, app_id, state_control, restart=False):

        minion_id = 'minion_{}'.format(app_id)
        stdout_log = os.path.join(self.log_dir, minion_id + '_out.log')
        stderr_log = os.path.join(self.log_dir, minion_id + '_err.log')
        log.debug('Forking minion %s.', minion_id)

        # Expires in 30 seconds and sets only if it doesn't exist
        state_control.set_minion_status(app_id, self.STARTED, ex=30, nx=False)

        # Setup command and launch the minion script. We return the subprocess
        # created as part of an active minion.
        open_opts = ['nohup', sys.executable, self.minion_executable,
                '-w', workflow_id, '-a', app_id, '-c', self.config_file_path]
        return subprocess.Popen(open_opts,
                stdout=open(stdout_log, 'a'), stderr=open(stderr_log, 'a'))

    def _terminate_minion(self, app_id):
        # In this case we got a request for terminating this workflow
        # execution instance (app). Thus, we are going to explicitly
        # terminate the workflow, clear any remaining metadata and return
        assert app_id in self.active_minions
        log.info("Received termination pill for app %s", app_id)
        os.kill(self.active_minions[app_id].pid, signal.SIGTERM)
        del self.active_minions[app_id]

    def minion_support(self):
        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port)
        while True:
            self.read_minion_support_queue(redis_conn)

    def read_minion_support_queue(self, redis_conn):
        try:
            state_control = StateControlRedis(redis_conn)
            ticket = json.loads(state_control.pop_master_queue())
            workflow_id = ticket.get('workflow_id')
            app_id = ticket.get('app_id')
            reason = ticket.get('reason')
            log.info("Master received a ticket for app %s", app_id)
            if reason == self.HELP_UNHANDLED_EXCEPTION:
                # Let's kill the minion and start another
                minion_info = json.loads(
                    state_control.get_minion_status(app_id))
                while True:
                    try:
                        os.kill(minion_info['pid'], signal.SIGKILL)
                    except OSError as err:
                        if err.errno == errno.ESRCH:
                            break
                    time.sleep(.5)

                self._start_minion(workflow_id, app_id, state_control)

            elif reason == self.HELP_STATE_LOST:
                pass
            else:
                log.warn("Unknown help reason %s", reason)

        except Exception as ex:
            log.error(ex)

    def watch_minion_status(self):
        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port)
        JuicerServer.watch_minion_process(redis_conn)

    @staticmethod
    def watch_minion_process(redis_conn):
        pubsub = redis_conn.pubsub()
        pubsub.psubscribe('__keyevent@*__:expired')
        for msg in pubsub.listen():
            if msg.get('type') == 'pmessage' and 'minion' in msg.get('data'):
                log.warn('Minion {id} stopped'.format(id=msg.get('data')))

    def process(self):
        self.start_process = multiprocessing.Process(
            name="master", target=self.start)
        self.start_process.daemon = False

        self.minion_support_process = multiprocessing.Process(
            name="help_desk", target=self.minion_support)
        self.minion_support_process.daemon = False

        self.minion_status_process = multiprocessing.Process(
            name="minion_status", target=self.watch_minion_status)
        self.minion_status_process.daemon = False

        self.start_process.start()
        self.minion_support_process.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--port", help="Listen port")
    parser.add_argument("-c", "--config", help="Config file", required=True)
    args = parser.parse_args()

    with open(args.config) as config_file:
        juicer_config = yaml.load(config_file.read())

    # Every minion starts with the same script.
    minion_executable = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'minion.py'))

    log.info('Starting Juicer Server')
    server = JuicerServer(juicer_config, minion_executable,
            config_file_path=args.config)
    server.process()
