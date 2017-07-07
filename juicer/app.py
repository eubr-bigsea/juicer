# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import print_function

import argparse
import json
import logging.config
from collections import defaultdict

import redis
import requests
import yaml
from juicer.runner import configuration
from juicer.service import limonero_service
from juicer.spark.transpiler import SparkTranspiler
from juicer.workflow.workflow import Workflow

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
    def __init__(self, redis_conn, workflow_id, execute_main, params, job_id,
                 config):
        self.redis_conn = redis_conn
        self.config = config
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

    def run(self):
        log.debug('Processing workflow queue %s', self.workflow_id)
        # msg = self.redis_conn.brpop(str(self.workflow_id))

        # self.redis_conn.hset(_id, 'status', Statuses.RUNNING)
        tahiti_conf = self.config['juicer']['services']['tahiti']

        r = requests.get(
            "{url}/workflows/{id}?token={token}".format(
                id=self.workflow_id, url=tahiti_conf['url'],
                token=tahiti_conf['auth_token']))
        loader = None
        if r.status_code == 200:
            loader = Workflow(json.loads(r.text), self.config)
        else:
            print(tahiti_conf['url'], r.text)
            exit(-1)
        # FIXME: Implement validation
        # loader.verify_workflow()
        configuration.set_config(self.config)
        spark_instance = SparkTranspiler(configuration.get_config())
        self.params['execute_main'] = self.execute_main

        # generated = StringIO()
        # spark_instance.output = generated
        '''
        try:
            spark_instance.transpile(loader.workflow,
                                     loader.graph,
                                     params=self.params,
                                     job_id=self.job_id)
        except ValueError as ve:
            log.exception("At least one parameter is missing", exc_info=ve)
        except:
            raise
        '''

        limonero_config = self.config['juicer']['services']['limonero']
        data_sources = []
        for t in loader.workflow['tasks']:
            if t['operation']['slug'] == 'data-reader':
                data_sources.append(limonero_service.query_limonero(
                    limonero_config['url'], '/datasources/',
                    str(limonero_config['auth_token']),
                    t['forms']['data_source']['value']))

        privacy_info = {}
        attribute_group_set = defaultdict(list)
        for ds in data_sources:
            attrs = []
            privacy_info[ds['id']] = {'attributes': attrs}
            for attr in ds['attributes']:
                privacy = attr.get('attribute_privacy', {}) or {}
                attribute_privacy_group_id = privacy.get(
                    'attribute_privacy_group_id')
                privacy_config = {
                    'id': attr['id'],
                    'name': attr['name'],
                    'type': attr['type'],
                    'privacy_type': privacy.get('privacy_type'),
                    'anonymization_technique': privacy.get(
                        'anonymization_technique'),
                    'attribute_privacy_group_id': attribute_privacy_group_id
                }
                attrs.append(privacy_config)
                if attribute_privacy_group_id:
                    attribute_group_set[attribute_privacy_group_id].append(
                        privacy_config)
                    # print('#' * 40)
                    # print(attr.get('name'), attr.get('type'))
                    # print(privacy.get('privacy_type'),
                    #       privacy.get('anonymization_technique'),
                    #       privacy.get('attribute_privacy_group_id'))

        anonymization_techniques = {
            'NO_TECHNIQUE': 0,
            'GENERALIZATION': 1,
            'MASK': 2,
            'ENCRYPTION': 3,
            'SUPPRESSION': 4
        }

        def sort_attr_privacy(a):
            return anonymization_techniques[a.get(
                'anonymization_technique', 'NO_TECHNIQUE')]

        for attributes in attribute_group_set.values():
            more_restritive = sorted(
                attributes, key=sort_attr_privacy, reverse=True)[0]
            # print(json.dumps(more_restritive[0], indent=4))
            # Copy all privacy config from more restrictive one
            for attribute in attributes:
                attribute.update(more_restritive)

        print(json.dumps(privacy_info, indent=2))


def main(workflow_id, execute_main, params, job_id, config):
    redis_conn = redis.StrictRedis()
    service = JuicerSparkService(redis_conn, workflow_id, execute_main, params,
                                 job_id, config)
    service.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False,
                        help="Configuration file")

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

    main(args.workflow, args.execute_main, {"service": args.service},
         args.job_id, juicer_config)
    '''
    if True:
        app.run(debug=True, port=8000)
    else:
        wsgi.server(eventlet.listen(('', 8000)), app)
    '''
