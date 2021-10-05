# coding=utf-8
import os
import tempfile
import traceback
import zipfile
from urllib.parse import urlparse

import pyarrow as pa
import yaml
from redis import StrictRedis
from juicer.service import limonero_service, stand_service
from rq import get_current_job
from gettext import gettext

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

