# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import urlparse
import logging
import pdb

import redis
import requests
from juicer.runner.control import StateControlRedis
from six import StringIO

# eventlet.monkey_patch()
import json

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s,%(msecs)05.1f (%(funcName)s) %(message)s',
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class JuicerClient:
    def __init__(self, redis_conn,
            workflow_id, app_id, app_configs={}, terminate=False):
        self.redis_conn = redis_conn
        self.workflow_id = workflow_id
        self.app_id = app_id
        self.app_configs = app_configs
        self.terminate = terminate

    def run(self):
        log.debug('Processing workflow queue %s', self.workflow_id)
        state_control = StateControlRedis(self.redis_conn)
        request_url = "http://beta.ctweb.inweb.org.br/tahiti/workflows/{}?token=123456".format(self.workflow_id)
        workflow_json = requests.get(request_url).text

        app_submission = {
                "workflow_id": self.workflow_id,
                "app_id": self.app_id,
                "app_configs": self.app_configs,
                "terminate": "true" if self.terminate else "false"
                }

        print workflow_json
        workflow_dict = json.loads(workflow_json)
        app_submission['workflow'] = workflow_dict
        app_submission_json = json.dumps(app_submission)
        print app_submission
        print app_submission_json
        
        state_control.push_start_queue(json.dumps(app_submission))

def main(redisserver, workflow_id, app_id, appconfigs, terminate):
    # redis server parsing
    parsed_url = urlparse.urlparse(redisserver)
    redis_conn = redis.StrictRedis(host=parsed_url.hostname,
            port=parsed_url.port)

    # app configs (spark) 'config1=value1,config2=value2, ...'
    if appconfigs:
        app_configs = [kv.split("=") for kv in appconfigs.split("=")]
        app_configs = {kv[0]:kv[1] for kv in app_configs}
    else:
        app_configs = {}

    client = JuicerClient(redis_conn,
            workflow_id, app_id, app_configs, terminate)
    client.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-r", "--redis_server", default="redis://127.0.0.1:6379",
            help="Please inform the redis server url redis://<host>:<port>")
    
    parser.add_argument("-w", "--workflow_id", type=str, required=True,
            help="Workflow identification number")
    
    parser.add_argument("-a", "--app_id", type=str, required=True,
            help="Application identification number")
    
    parser.add_argument("-c", "--app_configs", type=str, default="",
            help="Key value configuration for the spark application")

    parser.add_argument("-t", "--terminate", type=bool, default=False,
            help="terminate the context identified by (workflowid,appid)")

    args = parser.parse_args()
    print args

    main(args.redis_server,
            args.workflow_id, args.app_id, args.app_configs, args.terminate)
