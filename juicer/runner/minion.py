import argparse
import urlparse

import redis
import yaml
from juicer.spark.spark_minion import SparkMinion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("-a", "--app_id", help="Job id", type=int,
                        required=True)
    parser.add_argument("-t", "--type", help="Execution engine",
                        required=False, default="SPARK")
    args = parser.parse_args()

    with open(args.config) as config_file:
        juicer_config = yaml.load(config_file.read())

    parsed_url = urlparse.urlparse(
        juicer_config['juicer']['servers']['redis_url'])
    redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                   port=parsed_url.port)
    if args.type == 'SPARK':
        # log.info('Starting Juicer Spark Minion')
        server = SparkMinion(redis_conn, args.app_id, juicer_config)
        server.process()
    else:
        raise ValueError(
            "{type} is not supported (yet!)".format(type=args.type))
