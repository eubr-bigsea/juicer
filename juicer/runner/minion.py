# coding=utf-8
from __future__ import absolute_import

import argparse
import gettext
import logging.config

import matplotlib
import os
import redis
import yaml
from future.moves.urllib.parse import urlparse
from juicer.compss.compss_minion import COMPSsMinion
from juicer.keras.keras_minion import KerasMinion
from juicer.spark.spark_minion import SparkMinion
from juicer.scikit_learn.scikit_learn_minion import ScikitLearnMinion

# Important!
# See https://stackoverflow.com/a/29172195/1646932
matplotlib.use('Agg', force=True, warn=True)

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger(__name__)

# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Config file.", required=True)
    parser.add_argument("-w", "--workflow_id", help="Workflow id.", type=str,
                        required=True)
    parser.add_argument("-a", "--app_id", help="Job id", type=str,
                        required=False)
    parser.add_argument("-t", "--type", help="Execution engine.",
                        required=False, default="spark")
    parser.add_argument("--lang", help="Minion messages language (i18n).",
                        required=False, default="en_US")
    parser.add_argument("--jars", help="Add Java JAR files to class path.",
                        required=False)

    args = parser.parse_args()
    t = gettext.translation('messages', locales_path, [args.lang],
                            fallback=True)
    t.install(unicode=True)

    log.info(_("Starting minion"))
    log.debug(_('(c) Lemonade - DCC UFMG'))
    try:
        with open(args.config) as config_file:
            juicer_config = yaml.load(config_file.read())

        parsed_url = urlparse(
            juicer_config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port,
                                       decode_responses=True)
        if args.type == 'spark':
            # log.info('Starting Juicer Spark Minion')
            minion = SparkMinion(redis_conn,
                                 args.workflow_id,
                                 args.app_id or args.workflow_id,
                                 juicer_config,
                                 args.lang, args.jars)
        elif args.type == 'compss':
            # log.info('Starting COMPSs Minion')
            minion = COMPSsMinion(redis_conn,
                                  args.workflow_id,
                                  args.app_id or args.workflow_id,
                                  juicer_config,
                                  args.lang)
        elif args.type == 'scikit-learn':
            # log.info('Starting Scikit-learn Minion')
            minion = ScikitLearnMinion(redis_conn,
                                       args.workflow_id,
                                       args.app_id or args.workflow_id,
                                       juicer_config,
                                       args.lang)
        elif args.type == 'keras':
            log.info('Starting Keras Minion')
            minion = KerasMinion(redis_conn,
                                 args.workflow_id,
                                 args.app_id or args.workflow_id,
                                 juicer_config,
                                 args.lang)
        else:
            raise ValueError(
                _("{type} is not supported (yet!)").format(type=args.type))
        minion.process()
    except Exception as ex:
        log.exception(_("Error running minion"), exc_info=ex)
