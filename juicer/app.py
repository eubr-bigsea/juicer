# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import logging.config;
import pdb

import redis
import requests
from juicer.runner import configuration
from juicer.spark.transpiler import SparkTranspiler
from juicer.compss.transpiler import COMPSsTranspiler
from juicer.workflow.workflow import Workflow
from six import StringIO

import json
import yaml

logging.config.fileConfig('logging_config.ini')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Statuses:
    def __init__(self):
        pass

    EMPTY = 'EMPTY'
    START = 'START'
    RUNNING = 'RUNNING'


class JuicerSparkService:
    def __init__(self, redis_conn, workflow_id, execute_main, params, job_id):
        self.redis_conn = redis_conn
        self.workflow_id = workflow_id
        self.state = "LOADING"
        self.params = params
        self.job_id = job_id
        self.execute_main = execute_main
        self.states = {
            "EMPTY": {
                "START": self.start
            },
            "START": {

            }
        }

    def start(self):
        pass

    @staticmethod
    def get_operations():
        url = 'http://beta.ctweb.inweb.org.br/tahiti/operations?token=123456'
        r = requests.get(url)
        ops = json.loads(r.text)
        result = {}
        for op in ops:
            result[op['id']] = op
        return result

    def run(self):
        _id = 'status_{}'.format(self.workflow_id)
        # status = self.redis_conn.hgetall(_id)
        # print '>>>', status

        log.debug('Processing workflow queue %s', self.workflow_id)
        while True:
            # msg = self.redis_conn.brpop(str(self.workflow_id))

            # self.redis_conn.hset(_id, 'status', Statuses.RUNNING)

            r = requests.get(
                "http://beta.ctweb.inweb.org.br/tahiti/workflows/{}"
                "?token=123456".format(self.workflow_id))

            loader = Workflow(json.loads(r.text))
            #----- To test workflows in COMPSs
            with open('/home/lucasmsp/workspace/BigSea/testes_juicer/Workflow_TextTransformations_21848/w21848_Text.json') as json_data:
                r = json.load(json_data)
            #----- To test workflows in COMPSs

            loader = Workflow(r)
            # FIXME: Implement validation
            # loader.verify_workflow()

            if loader.plataform == "spark":
                spark_instance = SparkTranspiler(configuration)
                self.params['execute_main'] = self.execute_main

                # generated = StringIO()
                # spark_instance.output = generated
                try:
                    spark_instance.transpile(loader.workflow,
                            loader.graph,
                            params=self.params,
                            job_id=self.job_id)
                except ValueError as ve:
                    log.exception("At least one parameter is missing", exc_info=ve)
                    break
                except:
                    raise
            elif loader.plataform  == "compss":
                compss_instance = COMPSsTranspiler(loader.workflow,
                                                   loader.graph,
                                                   params=self.params)
                compss_instance.execute_main = self.execute_main

                # generated = StringIO()
                # spark_instance.output = generated
                try:
                    compss_instance.transpile()
                except ValueError as ve:
                    log.exception("At least one parameter is missing", exc_info=ve)
                    break
                except:
                    raise

            # generated.seek(0)
            # print generated.read()
            # raw_input('Pressione ENTER')
            break


def main(workflow_id, execute_main, params, job_id):
    redis_conn = redis.StrictRedis()
    service = JuicerSparkService(redis_conn, workflow_id, execute_main, params,
            job_id)
    service.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config", type=str, required=False, help="Configuration file")

    parser.add_argument("-w", "--workflow", type=int, required=True,
                        help="Workflow identification number")
    
    parser.add_argument("-j", "--job_id", type=int,
                        help="Job identification number")

    parser.add_argument("-e", "--execute-main", action="store_true",
                        help="Write code to run the program (it calls main()")

    parser.add_argument("-s", "--service", required=False,
                        action="store_true",
                        help="Indicates if workflow will run as a service")
    args = parser.parse_args()

    juicer_config = {}
    if args.config:
        with open(args.config) as config_file:
            juicer_config = yaml.load(config_file.read())

    configuration.set_config(juicer_config)

    main(args.workflow, args.execute_main, {"service": args.service},
            args.job_id)
    '''
    if True:
        app.run(debug=True, port=8000)
    else:
        wsgi.server(eventlet.listen(('', 8000)), app)
    '''
