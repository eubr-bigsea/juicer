# from project.database import db as _db
from __future__ import absolute_import

import gettext
import gzip
import json
import logging.config
import os
import sys

import pytest
from pyspark.sql import SparkSession

from juicer.spark.data_operation import DataReaderOperation

sys.path.append(os.path.dirname(os.path.curdir))


def pytest_sessionstart(session):
    locales_path = os.path.join(os.path.dirname(__file__), '..', 'juicer',
                                'i18n', 'locales')
    t = gettext.translation('messages', locales_path, ['en'],
                            fallback=True)
    t.install()
    logging.config.fileConfig('logging_config.ini')


# Mock for Limonero services
def patched_get_data_source_info(base_url, token, data_source_id):
    return {
        'attributes': [],
        'format': 'CSV',
        'url': 'http://hdfs.lemonade:9000'
    }


@pytest.fixture(scope='function')
def app(request):
    """Session-wide test `Flask` application."""
    settings_override = {
        'TESTING': True,
    }
    yield settings_override


@pytest.fixture(scope='session')
def spark_session():
    session = SparkSession.builder.master("local").appName(
        "Juicer").getOrCreate()
    yield session
    session.stop()


@pytest.fixture(scope='session')
def spark_transpiler(juicer_config_for_spark):
    from juicer.spark.transpiler import SparkTranspiler
    return SparkTranspiler(juicer_config_for_spark)


@pytest.fixture(scope='session')
def juicer_config_for_spark():
    return {
        'juicer': {
            'services': {

            }
        }
    }


@pytest.fixture(scope='session')
def spark_operations():
    path = 'tests/spark/fixtures/operations.json.gz'
    with gzip.open(path) as json_ops:
        return json.loads(json_ops.read())


@pytest.fixture(scope='session')
def iris_data():
    return {
        'url': 'file://{}/tests/spark/integration/iris.csv'.format(
            os.path.abspath(".")),
        'format': 'CSV',
        'is_first_line_header': 1,
        'attributes': [
            {'name': 'Sepal_length', 'type': "DECIMAL", 'precision': 10,
             'scale': 2},
            {'name': 'Sepal_width', 'type': "DECIMAL", 'precision': 10,
             'scale': 2},
            {'name': 'Petal_length', 'type': "DECIMAL", 'precision': 10,
             'scale': 2},
            {'name': 'Petal_width', 'type': "DECIMAL", 'precision': 10,
             'scale': 2},
            {'name': 'Species', 'type': "CHARACTER"},
        ]
    }


@pytest.fixture(scope='session')
def iris_workflow():
    return {
        'id': 1,
        'name': 'Data Reader test',
        'tasks': [
            {
                'id': '001',
                'operation': {
                    'id': 1, 'slug': DataReaderOperation.SLUG,
                },
                'forms': {
                    'display_sample': {'category': 'EXECUTION', 'value': 1},
                    DataReaderOperation.DATA_SOURCE_ID_PARAM:
                        {'category': 'EXECUTION', 'value': 1},
                }
            }
        ],
        'flows': [],
        'user': {'id': 1, 'name': 'Tester'}
    }


@pytest.fixture(scope='session')
def iris_data_frame(spark_session):
    url = 'file://{}/tests/spark/integration/iris.csv'.format(
        os.path.abspath("."))

    return spark_session.read.option('nullValue', '').option(
        'treatEmptyValuesAsNulls', 'true').option(
        'wholeFile', 'true').option(
        'multiLine', 'true').option('escape', '"').csv(
        url, quote=None, ignoreTrailingWhiteSpace=True, encoding='UTF-8',
        header=True, sep=',', inferSchema=True, mode='FAILFAST')
