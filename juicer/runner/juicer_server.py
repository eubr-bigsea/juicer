# coding=utf-8
"""
"
"""
import argparse
import json
import logging
import multiprocessing
import subprocess
import sys

import os
import yaml
from juicer.exceptions import JuicerException

logging.basicConfig(
    format=('[%(levelname)s] %(asctime)s,%(msecs)05.1f '
            '(%(funcName)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class StateControlRedis:
    """
    Controls state of Workflows, Minions and Jobs in Lemonade.
    For minions, it is important to know if they are running or not and which
    state they keep.
    For workflows, it is important to avoid running them twice.
    Job queue is used to control which commands minions should execute.
    Finally, job output queues contains messages from minions to be sent to
    user interface.
    """
    START_QUEUE_NAME = 'queue_start'

    def __init__(self, redis_conn):
        self.redis_conn = redis_conn

    def pop_start_queue(self, block=True):
        if block:
            result = self.redis_conn.blpop(self.START_QUEUE_NAME)[1]
        else:
            result = self.redis_conn.lpop(self.START_QUEUE_NAME)
        return result

    def push_start_queue(self, data):
        self.redis_conn.rpush(self.START_QUEUE_NAME, data)

    def pop_job_queue(self, job_id, block=True):
        key = 'queue_job:{}'.format(job_id)
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result

    def push_job_queue(self, job_id, data):
        key = 'queue_job:{}'.format(job_id)
        self.redis_conn.rpush(key, data)

    def get_job_queue_size(self, job_id):
        key = 'queue_job:{}'.format(job_id)
        return self.redis_conn.llen(key)

    def get_workflow_status(self, workflow_id):
        key = 'record_workflow:{}'.format(workflow_id)
        return self.redis_conn.hget(key, 'status')

    def set_workflow_status(self, workflow_id, status):
        key = 'record_workflow:{}'.format(workflow_id)
        self.redis_conn.hset(key, 'status', status)

    def get_workflow_data(self, workflow_id):
        key = 'record_workflow:{}'.format(workflow_id)
        return self.redis_conn.hgetall(key)

    def get_minion_status(self, job_id):
        key = 'key_minion_job:{}'.format(job_id)
        return self.redis_conn.get(key)

    def set_minion_status(self, job_id, status, ex=120, nx=True):
        key = 'key_minion_job:{}'.format(job_id)
        return self.redis_conn.set(key, status, ex=ex, nx=nx)

    def pop_job_output_queue(self, job_id, block=True):
        key = 'queue_output_job:{job_id}'.format(job_id=job_id)
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result

    def push_job_output_queue(self, job_id, data):
        key = 'queue_output_job:{job_id}'.format(job_id=job_id)
        self.redis_conn.rpush(key, data)

    def get_job_output_queue_size(self, job_id):
        key = 'queue_output_job:{job_id}'.format(job_id=job_id)
        return self.redis_conn.llen(key)


class JuicerServer:
    """
    Server
    """
    STARTED = 'STARTED'
    LOADED = 'LOADED'

    def __init__(self, config):
        self.queue = multiprocessing.Queue()
        self.consumer_process = None
        self.executor_process = None
        self.start_process = None
        self.config = config

    # def consume(self):
    #     filename = '/mnt/spark/teste.py'
    #     cached_stamp = os.stat(filename).st_mtime
    #     p = multiprocessing.current_process()
    #     # print 'Starting:', p.name, p.pid
    #     sys.stdout.flush()
    #     while True:
    #         stamp = os.stat(filename).st_mtime
    #         if stamp != cached_stamp:
    #             cached_stamp = stamp
    #             time.sleep(2)
    #             self.queue.put(filename)
    #     time.sleep(2)
    #     # print 'Exiting :', p.name, p.pid
    #     sys.stdout.flush()

    # def execute(self):
    #     p = multiprocessing.current_process()
    #     # print 'Starting:', p.name, p.pid
    #     sys.stdout.flush()
    #     modules = {}
    #     while True:
    #         filepath = self.queue.get()
    #         path = os.path.dirname(filepath)
    #         name = os.path.basename(filepath)
    #
    #         m = modules.get(filepath)
    #         if m is None:
    #             if path not in sys.path:
    #                 sys.path.append(path)
    #             print 'loading'
    #             m = importlib.import_module(name.split('.')[0])
    #             modules[filepath] = m
    #         else:
    #             print 'Reloading'
    #             reload(m)
    #
    #         print 'Starting processing'
    #         m.main()
    #         print 'Finished processing'
    #
    #     print 'Exiting :', p.name, p.pid
    #     sys.stdout.flush()

    # def start(self):
    #     log.info('Starting master process. Reading "start" queue ')
    #     parsed_url = urlparse.urlparse(
    #         self.config['juicer']['servers']['redis_url'])
    #     redis_con = redis.StrictRedis(host=parsed_url.hostname,
    #                                   port=parsed_url.port)
    #     minions_executable = os.path.abspath(
    #         os.path.join(os.path.dirname(__file__), 'minion.py'))
    #     log_dir = self.config['juicer'].get('log', {}).get('path', '/tmp')
    #
    #     while True:
    #         self.read_start_queue(log_dir, minions_executable, redis_con)

    # noinspection PyMethodMayBeStatic
    def read_start_queue(self, log_dir, minions_executable, redis_conn):

        state_control = StateControlRedis(redis_conn)
        msg = state_control.pop_start_queue()

        item = json.loads(msg)
        job_id = item.get('job_id')
        try:
            if job_id is None:
                raise ValueError('Job id not informed')
            minion_id = 'minion_{}'.format(job_id)
            stdout_log = os.path.join(log_dir, minion_id + '_out.log')
            stderr_log = os.path.join(log_dir, minion_id + '_err.log')

            if state_control.get_workflow_status(
                    item['workflow_id']) == JuicerServer.STARTED:
                raise JuicerException('Workflow is already started', code=1000)
            # Is minioon running?
            minion_info = state_control.get_minion_status(job_id)

            if minion_info:
                log.debug('Minion %s is running.', minion_id)
            else:
                log.debug('Forking minion %s.', minion_id)
                # Expires in 120 seconds and sets only if it doesn't exist
                state_control.set_minion_status(
                    job_id, self.STARTED, ex=120, nx=True)
                open_opts = ['nohup', sys.executable, minions_executable,
                             '-j', job_id]
                p = subprocess.Popen(
                    open_opts, stdout=open(stdout_log, 'a'),
                    stderr=open(stderr_log, 'a'))  # , preexec_fn=os.setpgrp)

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

    def process(self):
        # self.consumer_process = multiprocessing.Process(name="consumer",
        #                                                 target=self.consume)
        # self.executor_process = multiprocessing.Process(name="executor",
        #                                                 target=self.execute)
        # self.consumer_process.daemon = False
        # self.executor_process.daemon = False
        # self.consumer_process.start()
        # self.executor_process.start()
        self.start_process = multiprocessing.Process(
            name="master", target=self.read_start_queue)
        self.start_process.daemon = False
        self.start_process.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--port", help="Listen port")
    parser.add_argument("-c", "--config", help="Config file", required=True)
    args = parser.parse_args()

    with open(args.config) as config_file:
        juicer_config = yaml.load(config_file.read())

    log.info('Starting Juicer Server')
    server = JuicerServer(juicer_config)
    server.process()
