# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import pdb

import redis
import requests
from juicer.spark.transpiler import SparkTranspiler
from juicer.workflow.workflow import Workflow
from six import StringIO

# eventlet.monkey_patch()
import json

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s,%(msecs)05.1f (%(funcName)s) %(message)s',
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Statuses:
    def __init__(self):
        pass

    EMPTY = 'EMPTY'
    START = 'START'
    RUNNING = 'RUNNING'


class JuicerSparkService:
    def __init__(self, redis_conn, workflow_id, execute_main, params):
        self.redis_conn = redis_conn
        self.workflow_id = workflow_id
        self.state = "LOADING"
        self.params = params
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
            # FIXME: Implement validation
            # loader.verify_workflow()
            spark_instance = SparkTranspiler()
            spark_instance.execute_main = self.execute_main

            # generated = StringIO()
            # spark_instance.output = generated
            try:
                spark_instance.transpile(loader.workflow, loader.graph,
                                         params=self.params)
            except ValueError as ve:
                log.exception("At least one parameter is missing", exc_info=ve)
                break
            except:
                raise

            # generated.seek(0)
            # print generated.read()
            # raw_input('Pressione ENTER')
            break


def main(workflow_id, execute_main, params):
    redis_conn = redis.StrictRedis()
    service = JuicerSparkService(redis_conn, workflow_id, execute_main, params)
    service.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", type=str, help="Configuration file")
    parser.add_argument("-w", "--workflow", type=int, required=True,
                        help="Workflow identification number")
    parser.add_argument("-e", "--execute-main", action="store_true",
                        help="Write code to run the program (it calls main()")

    parser.add_argument("-s", "--service", required=False,
                        action="store_true",
                        help="Indicates if workflow will run as a service")
    args = parser.parse_args()

    main(args.workflow, args.execute_main, {"service": args.service})
    '''
    if True:
        app.run(debug=True, port=8000)
    else:
        wsgi.server(eventlet.listen(('', 8000)), app)
    '''
