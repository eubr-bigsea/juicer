# coding=utf-8
import os
import tempfile
from urllib.parse import urlparse
import traceback
import pyarrow as pa
from juicer.atmosphere import opt_ic
from redis import StrictRedis
import json
from juicer.service import limonero_service
import yaml

class RedisConn(StrictRedis):
    def __init__(self, host='localhost', port=6379, db=0, password=None, socket_timeout=None, 
            socket_connect_timeout=None, socket_keepalive=None, socket_keepalive_options=None, 
            connection_pool=None, unix_socket_path=None, encoding='utf-8', 
            encoding_errors='strict', charset=None, errors=None, decode_responses=False, 
            retry_on_timeout=False, ssl=False, ssl_keyfile=None, ssl_certfile=None, 
            ssl_cert_reqs='required', ssl_ca_certs=None, max_connections=None, 
            single_connection_client=False, health_check_interval=0):
        print('==============> Lemonade')
        print(host, port, db, password)
        super().__init__(decode_responses=False)
        # super().__init__(host=host, port=port, db=db, password=password, 
        #         socket_timeout=socket_timeout, 
        #         connection_pool=connection_pool, unix_socket_path, encoding, 
        #         encoding_errors, charset, errors, decode_responses=True,
        #         retry_on_timeout, ssl, ssl_keyfile, ssl_certfile,
        #         ssl_cert_reqs, ssl_ca_certs, max_connections, 
        #         single_connection_client, health_check_interval
        #         )

def estimate_time_with_performance_model(payload):
    # retrieves the model
    print(payload)
    model_id = payload['model_id']
    config_file = os.environ.get('JUICER_CONF')
    if config_file == None:
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

