import gc
import argparse
import importlib
import json
import logging

import time

import datetime
import sys

import os
import urlparse

import multiprocessing

import redis
import yaml
from juicer.runner.control import StateControlRedis
from juicer.util.string_importer import StringImporter

logging.basicConfig(
    format=('[%(levelname)s] %(asctime)s,%(msecs)05.1f '
            '(%(funcName)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Minion:
    def __init__(self, config, job_id):
        self.config = config
        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        self.redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                            port=parsed_url.port)
        self.state_control = StateControlRedis(self.redis_conn)
        self.job_id = job_id

    def process(self):
        raise NotImplementedError()


class CompssMinion(Minion):
    def process(self):
        pass


class SparkMinion(Minion):
    def __init__(self, config, job_id):
        Minion.__init__(self, config, job_id)
        self.start_process = None
        self.ping_process = None
        self.string_importer = StringImporter()
        sys.meta_path.append(self.string_importer)

    def ping(self):
        """ Pings redis to inform master this minion is online """
        log.info('Start ping')
        while True:
            status = {
                'status': 'READY',
                'pid': os.getpid(),
            }
            self.state_control.set_minion_status(self.job_id,
                                                 json.dumps(status), nx=False)
            time.sleep(5)

    def generate_output(self, msg):
        m = json.dumps({'message': msg, 'job_id': self.job_id,
                        'date': datetime.datetime.now().isoformat()})
        self.state_control.push_job_output_queue(self.job_id, m)

    def start(self):
        while True:
            try:
                job_info = json.loads(
                    self.state_control.pop_job_queue(self.job_id))
                self.generate_output('Starting job {}'.format(self.job_id))
                code = job_info.get('code')
                if code:
                    # Hot swap of code
                    module_name = 'juicer_job_{}'.format(self.job_id)
                    module = self.string_importer.add_or_update_module(
                        module_name, code)
                    module.main()
                    log.debug('Objects in memory after loading module: %s',
                              len(gc.get_objects()))
                else:
                    msg = 'No code was passed to the minion'
                    log.warn(msg)
                    self.generate_output(msg)
            except ValueError as ve:
                msg = 'Invalid message format: {}'.format(ve.message)
                log.warn(msg)
                self.generate_output(msg)
            except SyntaxError as se:
                msg = 'Invalid Python code: {}'.format(se)
                log.warn(msg)
                self.generate_output(msg)

    def process(self):
        self.start()
        return
        self.start_process = multiprocessing.Process(
            name="minion", target=self.start)
        self.start_process.daemon = False

        self.ping_process = multiprocessing.Process(
            name="ping process", target=self.ping)
        self.ping_process.daemon = False

        self.start_process.start()
        self.ping_process.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("-j", "--job_id", help="Job id", type=int,
                        required=True)
    parser.add_argument("-t", "--type", help="Processing technology type",
                        required=False, default="SPARK")
    args = parser.parse_args()

    with open(args.config) as config_file:
        juicer_config = yaml.load(config_file.read())

    if args.type == 'SPARK':
        log.info('Starting Juicer Spark Minion')
        server = SparkMinion(juicer_config, args.job_id)
        server.process()
    else:
        raise ValueError(
            "{type} is not supported (yet!)".format(type=args.type))
