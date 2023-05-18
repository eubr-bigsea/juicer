# coding=utf-8
import os
import json
import tempfile
import traceback
import zipfile
from urllib.parse import urlparse
from gettext import gettext
import pyarrow as pa
import yaml
import logging.config
from redis import StrictRedis
from juicer.service import limonero_service, stand_service
from rq import get_current_job
from gettext import gettext


logging.config.fileConfig('logging_config.ini') 
log = logging.getLogger('juicer.jobs')

def get_config():
    config_file = os.environ.get('JUICER_CONF')
    if config_file is None:
        result = {
            'status': 'ERROR',
            'message': gettext('You must inform the JUICER_CONF env variable')
        }
        return result

    with open(config_file) as f:
        config = yaml.load(f)
    return config


class JuicerStrictRedis(StrictRedis):
    def __init__(self, *args, **kwargs):
        config_file = os.environ.get('JUICER_CONF')
        if config_file is None:
            raise ValueError(
                    'You must inform the JUICER_CONF env variable')
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        redis_url = config['juicer']['servers']['redis_url']
        parsed = urlparse(redis_url)
        print(redis_url)
        super(JuicerStrictRedis, self).__init__(
                host=parsed.hostname, port=parsed.port)

def convert_data_source(payload):
    try:
        source_id = payload.get('id')
        
        if not source_id:
            raise ValueError(gettext('Invalid data source id'))
        config = get_config()
        
        limonero_config = config['juicer']['services']['limonero']
        token = str(limonero_config['auth_token'])
        ds = limonero_service.get_data_source_info(
            limonero_config['url'],
            token,
            source_id)

        valid_user = (ds.get('user_id') == payload.get('user', {}).get('id') or 
            'ADMINISTRATOR' in payload.get('user', {}).get('permissions', []))
        if not valid_user:
            raise ValueError(gettext('User does not own data source'))

        if not ds.get('enabled'):
            raise ValueError(gettext('Data source is not enabled and cannot be converted.'))

        if ds.get('format') != 'CSV':
            raise ValueError(gettext('Only CSV files can be converted to PARQUET'))

        if ds.get('storage', {}).get('type') != 'HDFS':
            raise ValueError(gettext('Only CSV files stored in HDFS can be converted'))

        from pyspark import SparkContext, SparkConf
        from pyspark.sql import SparkSession
        from pyspark.sql.types import StructType, StructField, IntegerType, DateType, StringType, FloatType, \
            BinaryType, DoubleType, TimestampType, DecimalType, LongType

        def get_spark_type(name_limonero):
            LIMONERO_TO_SPARK_DATA_TYPES = {
                "BINARY": BinaryType,
                "CHARACTER": StringType,
                "DATETIME": TimestampType,
                "DATE": DateType,
                "DOUBLE": DoubleType,
                "DECIMAL": DecimalType,
                "FLOAT": FloatType,
                "LONG": LongType,
                "INTEGER": IntegerType,
                "TEXT": StringType,
            }
            return LIMONERO_TO_SPARK_DATA_TYPES.get(name_limonero)()
           
        schema = StructType([
            StructField(c.get('name'), 
                        get_spark_type(c.get('type')),
                        True)
            for c in ds.get('attributes', [])
        ])
        # os.environ['HADOOP_USER_NAME'] = payload.get('user').get('name') 
        spark = SparkSession.builder \
            .master("local") \
            .appName("parquet_conversion") \
            .getOrCreate()

        df = spark.read.csv(ds.get('url'), 
                            header=ds.get('is_first_line_header'),
                            schema=schema)
        new_path = f'{ds.get("url")}.parquet'
        df.repartition(1).write.parquet(path=new_path,
                         mode="overwrite")
       
        base_url = limonero_config['url']
        payload = ds.copy()
        payload['storage_id'] = payload['storage']['id']
        payload['format'] = 'PARQUET'
        payload['name'] = f'{payload["name"]} - PARQUET'
        payload['url'] = new_path
        for attr in payload['attributes']:
            del attr['id']

        for permission in payload['permissions']:
            permission['id'] = 0
        

        resp = limonero_service.register_datasource(
            base_url, payload, token, mode='')

        return {'status': 'OK', 'message': 'Success', 'detail': {'id': resp.get('id')}}
    except Exception as e:
        log.error(e)
        return {'status': 'ERROR', 'message': gettext('Internal error'),
            'detail': str(e)}

