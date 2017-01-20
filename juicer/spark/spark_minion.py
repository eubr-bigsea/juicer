# coding=utf-8
import pyspark

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
    # Errors and messages
    MNN000 = ('MNN000', 'Success.')
    MNN001 = ('MNN001', 'Port output format not supported.')
    MNN002 = ('MNN002', 'Success getting data from task.')
    MNN003 = ('MNN003', 'State does not exists, processing job.')
    MNN004 = ('MNN004', 'Invalid port.')
    MNN005 = ('MNN005', 'Unable to retrieve data because a previous error.')
    MNN006 = ('MNN006', 'Invalid Python code or incorrect encoding: {}')

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

    def _generate_output(self, msg, status=None, code=None):
        """
        Sends feedback about execution of this minion.
        """
        obj = {'message': msg, 'job_id': self.job_id, 'code': code,
               'date': datetime.datetime.now().isoformat(),
               'status': status if status is not None else 'OK'}

        m = json.dumps(obj)
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
        result = True
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
                # Get rid of .pyc file if it exists
                if os.path.isfile('{}c'.format(generated_code_path)):
                    os.remove('{}c'.format(generated_code_path))
                # Hot swap of code
                self.module = imp.reload(self.module)

            self.state = self.module.main()
            log.debug('Objects in memory after loading module: %s',
                      len(gc.get_objects()))

        except UnicodeEncodeError as ude:
            msg = self.MNN006[1].format(ude)
            log.warn(msg)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False
        except ValueError as ve:
            msg = 'Invalid message format: {}'.format(ve.message)
            log.warn(msg)
            self._generate_output(msg, 'ERROR')
            result = False
        except SyntaxError as se:
            msg = self.MNN006[1].format(se)
            log.warn(msg)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False
        return result

    def deliver(self):
        """
        Process requests to deliver data processed by this minion.
        """
        while True:
            self._perform_deliver()

    @staticmethod
    def _convert_to_csv(row):
        result = []
        for v in row:
            t = type(v)
            if t in [datetime.datetime]:
                result.append(v.isoformat())
            elif t in [unicode, str]:
                result.append('"{}"'.format(v))
            else:
                result.append(str(v))
        return ','.join(result)

    def _send_to_output(self, data):
        self.state_control.push_job_output_queue(
            self.job_id, json.dumps(data))

    def _perform_deliver(self):

        request = json.loads(
            self.state_control.pop_job_delivery_queue(self.job_id))
        task_id = request['task_id']
        # FIXME: state must expire. How to identify this in the interface?
        # FIXME: Define how to identify the request.
        # FIXME: Define where to store generated data (Redis?)
        if task_id in self.state:
            self._read_dataframe_data(request, task_id)
        else:
            data = {'status': 'WARNING', 'code': self.MNN003[0],
                    'message': self.MNN003[1]}
            self._send_to_output(data)

            # FIXME: Report missing or process workflow until this task
            workflow = request['workflow']
            self.state_control.push_job_queue(self.job_id, workflow)
            if self._perform_execute():
                self._read_dataframe_data(request, task_id)
            else:
                data = {'status': 'ERROR', 'code': self.MNN005[0],
                        'message': self.MNN005[1]}
                self._send_to_output(data)

    def _read_dataframe_data(self, request, task_id):
        # Perform a collection action in data frame.
        # FIXME: Evaluate if there is a better way to identify the port
        port = int(request.get('port'))

        # Last position in state is the execution time, so it should be ignored
        if len(self.state[task_id]) - 1 >= port:
            df = self.state[task_id][port]

            # Evaluating if df has method "take" allows unit testing
            # instead of testing exact pyspark.sql.dataframe.Dataframe
            # type check.
            if df is not None and hasattr(df, 'take'):
                # FIXME define as a parameter?:
                result = df.take(100).rdd.map(
                    SparkMinion._convert_to_csv).collect()
                out_queue = request.get('output')
                self.state_control.push_queue(
                    out_queue, '\n'.join(result))
                data = {'status': 'SUCCESS', 'code': self.MNN002[0],
                        'message': self.MNN002[1], 'output': out_queue}
                self._send_to_output(data)
            else:
                data = {'status': 'ERROR', 'code': self.MNN001[0],
                        'message': self.MNN001[1]}
                self._send_to_output(data)
        else:
            data = {'status': 'ERROR', 'code': self.MNN004[0],
                    'message': self.MNN004[1]}
            self._send_to_output(data)

    def process(self):
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
