# coding=utf-8
"""
"
"""
import argparse
import importlib
import json
import logging
import multiprocessing
import sys
import time
import urlparse

from datetime import datetime

import os
import redis
import yaml
from juicer.runner.control import StateControlRedis

sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python'))

spark = importlib.import_module('spark')


logging.basicConfig(
    format=('[%(levelname)s] %(asctime)s,%(msecs)05.1f '
            '(%(funcName)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class JuicerMinion:
    """
    Minion
    """
    def __init__(self, config, job_id, spark_master):
        self.queue = multiprocessing.Queue()
        self.start_process = None
        self.ping_process = None
        self.config = config
        self.job_id = job_id

        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        self.redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                            port=parsed_url.port)
        self.spark_master = spark_master

    def start(self):
        log.info('Starting master process. Reading "start" queue ')

        minions_executable = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'minion.py'))
        log_dir = self.config['juicer'].get('log', {}).get('path', '/tmp')

        while True:
            self.process_start(log_dir, minions_executable)

    # noinspection PyMethodMayBeStatic
    def process_start(self, log_dir, minions_executable):

        state_control = StateControlRedis(self.redis_conn)
        workflow = json.loads(state_control.pop_job_queue(self.job_id))

        # conf = SparkConf()
        # conf.setMaster(self.spark_master)
        # conf.setAppName(workflow['name'])

        print 'Read', workflow

    def ping(self):
        """
        Keeps pinging server while minion is alive.
        """
        state_control = StateControlRedis(self.redis_conn)
        while True:
            log.info('Minion for job %s is pinging.', self.job_id)
            self.process_ping(state_control)
            time.sleep(3)

    def process_ping(self, state_control):
        status = {"status": "STARTED", 'pid': os.getpid(),
                  'date': datetime.now().isoformat()}
        state_control.set_minion_status(self.job_id, json.dumps(status),
                                        nx=False)

    def process(self):
        self.start_process = multiprocessing.Process(
            name="start", target=self.start)
        self.start_process.daemon = False

        self.ping_process = multiprocessing.Process(
            name="ping", target=self.ping)
        self.ping_process.daemon = False

        self.start_process.start()
        self.ping_process.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("-j", "--job", help="Job identifier", required=True)
    parser.add_argument("--spark-master", help="Spark Master", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        juicer_config = yaml.load(f.read())

    log.info('Starting Juicer Minion for Job #%s', args.job)
    minion = JuicerMinion(juicer_config, args.job, arg.spark_master)
    minion.process()
