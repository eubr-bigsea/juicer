import gc
import json
import multiprocessing
import sys
import time

import os
from juicer.runner.minion import Minion, log
from juicer.util.string_importer import StringImporter


class SparkMinion(Minion):
    """
    Controls the execution of Spark code in Lemonade Juicer.
    """
    def __init__(self, config, job_id):
        Minion.__init__(self, config, job_id)

        self.start_process = None
        self.ping_process = None
        self.delivery_process = None

        self.string_importer = StringImporter()
        self.state = None
        sys.meta_path.append(self.string_importer)

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
            status = {
                'status': 'READY',
                'pid': os.getpid(),
            }
            self.state_control.set_minion_status(self.job_id,
                                                 json.dumps(status), nx=False)
            time.sleep(5)

    def start(self):
        """
        Starts consuming jobs that must be processed by this minion.
        """
        while True:
            try:
                job_info = json.loads(
                    self.state_control.pop_job_queue(self.job_id))
                self._generate_output('Starting job {}'.format(self.job_id))
                code = job_info.get('code')
                if code:
                    # Hot swap of code
                    module_name = 'juicer_job_{}'.format(self.job_id)
                    module = self.string_importer.add_or_update_module(
                        module_name, code)
                    self.state = module.main()
                    log.debug('Objects in memory after loading module: %s',
                              len(gc.get_objects()))
                else:
                    msg = 'No code was passed to the minion'
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
        def convert_to_csv_field(v):
            import datetime
            t = type(v)
            if t in [datetime.datetime]:
                return v.isoformat()
            elif t in [unicode, str]:
                return '"{}"'.format(v)
            else:
                return str(v)

        while True:
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
                    lambda x: ",".join(map(convert_to_csv_field, x))).collect()
                self.state_control.push_queue(request.get('output'),
                                              '\n'.join(result))

            else:
                pass
                # FIXME: Report missing or process until this task

    def process(self):
        self.start()
        return
        self.start_process = multiprocessing.Process(
            name="minion", target=self.start)
        self.start_process.daemon = False

        self.ping_process = multiprocessing.Process(
            name="ping process", target=self.ping)
        self.ping_process.daemon = False

        self.delivery_process = multiprocessing.Process(
            name="delivery", target=self.deliver)
        self.start_process.daemon = False

        self.start_process.start()
        self.ping_process.start()
        self.delivery_process.start()
