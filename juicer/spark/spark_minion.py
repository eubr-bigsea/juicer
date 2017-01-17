# coding=utf-8
import codecs
import gc
import importlib
import json
import logging
import multiprocessing
import sys
import time
from io import StringIO

import datetime
import imp

import os
from juicer.runner.minion_base import Minion
from juicer.spark.transpiler import SparkTranspiler
from juicer.util.string_importer import StringImporter
from juicer.workflow.workflow import Workflow

log = logging.getLogger()
log.setLevel(logging.WARN)


class SparkMinion(Minion):
    """
    Controls the execution of Spark code in Lemonade Juicer.
    """

    def __init__(self, redis_conn, job_id, config):
        Minion.__init__(self, redis_conn, job_id, config)

        self.execute_process = None
        self.ping_process = None
        self.delivery_process = None
        self.module = None

        self.string_importer = StringImporter()
        self.state = None
        self.transpiler = SparkTranspiler()
        self.config = config
        sys.meta_path.append(self.string_importer)
        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')

        sys.path.append(self.tmp_dir)

    def _generate_output(self, msg):
        """
        Sends feedback about execution of this minion.
        """
        m = json.dumps({'message': msg, 'job_id': self.job_id,
                        'date': datetime.datetime.now().isoformat()})
        self.state_control.push_job_output_queue(self.job_id, m)

    def ping(self):
        """ Pings redis to inform master this minion is online """
        log.info('Start ping')
        while True:
            self._perform_ping()
            time.sleep(5)

    def _perform_ping(self):
        status = {
            'status': 'READY', 'pid': os.getpid(),
        }
        self.state_control.set_minion_status(self.job_id, json.dumps(status),
                                             nx=False)

    def execute(self):
        """
        Starts consuming jobs that must be processed by this minion.
        """
        while True:
            self._perform_execute()

    def _perform_execute(self):
        try:
            job_info = json.loads(
                self.state_control.pop_job_queue(self.job_id))
            self._generate_output('Starting job {}'.format(self.job_id))
            workflow = job_info.get('workflow')

            loader = Workflow(workflow)
            module_name = 'juicer_job_{}'.format(self.job_id)

            generated_code_path = os.path.join(
                self.tmp_dir, module_name + '.py')

            with codecs.open(generated_code_path, 'w', 'utf8') as out:
                self.transpiler.transpile(
                    loader.workflow, loader.graph, {}, out)

            if self.module is None:
                self.module = importlib.import_module(module_name)
            else:
                # Hot swap of code
                self.module = imp.reload(self.module)

            self.state = self.module.main()
            log.debug('Objects in memory after loading module: %s',
                      len(gc.get_objects()))

        except UnicodeEncodeError as ude:
            msg = 'Invalid encode error: {}'.format(ude)
            log.warn(msg)
            self._generate_output(msg)
        except ValueError as ve:
            msg = 'Invalid message format: {}'.format(ve.message)
            log.warn(msg)
            self._generate_output(msg)
        except SyntaxError as se:
            msg = 'Invalid Python code: {}'.format(se)
            log.warn(msg)
            self._generate_output(msg)

    def deliver(self):
        """
        Process requests to deliver data processed by this minion.
        """
        while True:
            self._perform_deliver()

    @staticmethod
    def _convert_to_csv_field(v):
        t = type(v)
        if t in [datetime.datetime]:
            return v.isoformat()
        elif t in [unicode, str]:
            return '"{}"'.format(v)
        else:
            return str(v)

    def _perform_deliver(self):
        request = json.loads(
            self.state_control.pop_job_delivery_queue(self.job_id))
        # FIXME: state must expire. How to identify this in the interface?
        # FIXME: Define how to identify the request.
        # FIXME: Define where to store generated data (Redis?)
        if request.get('task_id') in self.state:
            # Perform a collection action in data frame.
            # FIXME: Evaluate if there is a better way to identify the port
            port = int(request.get('port'))
            df = self.state[request['task_id'][port]]

            # FIXME define as a parameter?:
            result = df.take(100).rdd.map(
                lambda x: ",".join(
                    map(SparkMinion._convert_to_csv_field, x))).collect()
            self.state_control.push_queue(request.get('output'),
                                          '\n'.join(result))

        else:
            pass
            # FIXME: Report missing or process until this task

    def process(self):
        self.execute()
        return
        self.execute_process = multiprocessing.Process(
            name="minion", target=self.execute)
        self.execute_process.daemon = False

        self.ping_process = multiprocessing.Process(
            name="ping process", target=self.ping)
        self.ping_process.daemon = False

        self.delivery_process = multiprocessing.Process(
            name="delivery", target=self.deliver)
        self.delivery_process.daemon = False

        self.execute_process.start()
        self.ping_process.start()
        self.delivery_process.start()
