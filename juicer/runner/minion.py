import argparse
import gettext
import locale
import logging.config
import urlparse

import os
import redis
import yaml
from juicer.spark.spark_minion import SparkMinion

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger(__name__)

# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("-w", "--workflow_id", help="Workflow id", type=str,
                        required=True)
    parser.add_argument("-a", "--app_id", help="Job id", type=str,
                        required=False)
    parser.add_argument("-t", "--type", help="Execution engine",
                        required=False, default="SPARK")
    parser.add_argument("--lang", help="Minion messages language (i18n)",
                        required=False, default="en_US")
    args = parser.parse_args()

    t = gettext.translation('messages', locales_path, [args.lang],
                            fallback=True)
    t.install()

    log.info(_("Starting minion"))
    try:
        with open(args.config) as config_file:
            juicer_config = yaml.load(config_file.read())

        parsed_url = urlparse.urlparse(
            juicer_config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port)
        if args.type == 'SPARK':
            # log.info('Starting Juicer Spark Minion')
            server = SparkMinion(redis_conn,
                                 args.workflow_id,
                                 args.app_id or args.workflow_id, juicer_config)
            server.process()
        else:
            raise ValueError(_(
                "{type} is not supported (yet!)").format(type=args.type))
    except Exception as ex:
        log.exception(_("Error running minion"), exc_info=ex)
