# coding=utf-8
import os
import tempfile
import traceback
from urllib.parse import urlparse

import pyarrow as pa
import yaml
from juicer.atmosphere import opt_ic
from juicer.service import limonero_service


def estimate_time_with_performance_model(payload):
    # retrieves the model
    print(payload)
    model_id = payload['model_id']
    config_file = os.environ.get('JUICER_CONF')
    if config_file is None:
        result = {
            'status': 'ERROR',
            'message': 'You must inform the JUICER_CONF env variable'
        }
        return result
    with open(config_file) as f:
        config = yaml.load(f)
    limonero_conf = config['juicer']['services']['limonero']
    model = limonero_service.get_model_info(
        limonero_conf['url'], str(limonero_conf['auth_token']),
        model_id)

    storage = model['storage']
    cores = payload.get('cores', [2])
    try:
        if storage['type'] == 'HDFS':
            parsed = urlparse(storage['url'])
            if parsed.scheme == 'file':
                path = os.path.abspath(os.path.join(parsed.path, model['path']))
            else:
                path = os.path.join(tempfile.gettempdir(),
                                    os.path.basename(model['path']))
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    fs = pa.hdfs.connect(parsed.hostname, parsed.port)
                    with open(path, 'wb') as temp:
                        with fs.open(model['path'], 'rb') as f:
                            temp.write(f.read())
                    fs.close()

            optimizer = opt_ic.Optimizer(
                os.path.splitext(os.path.basename(path))[0],
                20, os.path.dirname(path))
            # Deadline is in seconds, model expects milliseconds
            deadline = int(payload.get('deadline', 3600))
            result = {
                'status': 'OK',
                'deadline': deadline,
                'cores': cores,
                'result': [
                    optimizer.solve(core, deadline)
                    for core in cores
                    ]
            }
        else:
            result = {
                'status': 'ERROR',
                'message': 'Unsupported storage: {}'.format(storage['type'])
            }
    except Exception as ex:
        result = {
            'status': 'ERROR',
            'message': 'Internal error: {}'.format(ex),
            'stack': traceback.format_exc()
        }
    return result
