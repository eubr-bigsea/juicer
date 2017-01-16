import argparse
import logging
import urlparse

import redis
import yaml
from juicer.runner.control import StateControlRedis
from juicer.spark.spark_minion import SparkMinion

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
