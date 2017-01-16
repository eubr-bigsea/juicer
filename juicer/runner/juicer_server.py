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
            '(%(funcName)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class JuicerServer:
    """
    Server
    """
    STARTED = 'STARTED'
    LOADED = 'LOADED'
    HELP_UNHANDLED_EXCEPTION = 1
    HELP_STATE_LOST = 2

    def __init__(self, config, minion_executable, log_dir='/tmp',
                 config_file_path=None):

        self.minion_support_process = None
        self.start_process = None
        self.minion_status_process = None

        self.config = config
        self.config_file_path = config_file_path
        self.minion_executable = minion_executable
        self.log_dir = log_dir

    def start(self):
        log.info('Starting master process. Reading "start" queue ')
        minions_executable = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'minion.py'))
        log_dir = self.config['juicer'].get('log', {}).get('path', '/tmp')

        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port)

        while True:
            self.read_start_queue(redis_conn)

    # noinspection PyMethodMayBeStatic
    def read_start_queue(self, redis_conn):

        state_control = StateControlRedis(redis_conn)
        msg = state_control.pop_start_queue()

        item = json.loads(msg)
        job_id = item.get('job_id')
        try:
            if job_id is None:
                raise ValueError('Job id not informed')
            minion_id = 'minion_{}'.format(job_id)

            if state_control.get_workflow_status(
                    item['workflow_id']) == JuicerServer.STARTED:
                raise JuicerException('Workflow is already started', code=1000)
            # Is minioon running?
            minion_info = state_control.get_minion_status(job_id)

            if minion_info:
                log.debug('Minion %s is running.', minion_id)
            else:
                self._start_minion(job_id, state_control)

            # requeue message to minion processing
            state_control.push_job_queue(job_id, msg)
            state_control.set_workflow_status(item['workflow_id'], self.STARTED)

            log.info('Generating code for job %s', job_id)
            state_control.push_job_output_queue(job_id, json.dumps(
                {'code': 0, 'message': 'Workflow will start soon'}))

        except JuicerException as je:
            log.error(je)
            if job_id:
                state_control.push_job_output_queue(job_id, json.dumps(
                    {'code': je.code, 'message': je.message}))
        except Exception as ex:
            log.error(ex)
            if job_id:
                state_control.push_job_output_queue(
                    job_id, json.dumps({'code': 500, 'message': ex.message}))
                # raise

    def _start_minion(self, job_id, state_control, restart=False):

        minion_id = 'minion_{}'.format(job_id)
        stdout_log = os.path.join(self.log_dir, minion_id + '_out.log')
        stderr_log = os.path.join(self.log_dir, minion_id + '_err.log')

        log.debug('Forking minion %s.', minion_id)

        # Expires in 30 seconds and sets only if it doesn't exist
        state_control.set_minion_status(job_id, self.STARTED, ex=30,
                                        nx=restart)
        open_opts = ['nohup', sys.executable, self.minion_executable,
                     '-j', job_id, '-c', self.config_file_path]
        subprocess.Popen(open_opts, stdout=open(stdout_log, 'a'),
                         stderr=open(stderr_log, 'a'))

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
            job_id = ticket.get('job_id')
            reason = ticket.get('reason')
            log.info("Master received a ticket for job %s", job_id)
            if reason == self.HELP_UNHANDLED_EXCEPTION:
                # Let's kill the minion and start another
                minion_info = json.loads(
                    state_control.get_minion_status(job_id))
                while True:
                    try:
                        os.kill(minion_info['pid'], signal.SIGKILL)
                    except OSError as err:
                        if err.errno == errno.ESRCH:
                            break
                    time.sleep(.5)

                self._start_minion(job_id, state_control)

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
                log.warn('Minion {id} stoped'.format(id=msg.get('data')))

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

    log.info('Starting Juicer Server')
    server = JuicerServer(juicer_config, args.config)
    server.process()
