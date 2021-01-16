# coding=utf-8
import os
import tempfile
import traceback
import zipfile
from urllib.parse import urlparse

import pyarrow as pa
import yaml
from redis import StrictRedis
from juicer.atmosphere import opt_ic, gpu_prediction
from juicer.atmosphere.vallum import perform_copy
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

def start_workflow(payload):
    """ Starts a workflow, creating a job """
    print(payload)

    return payload
    
def estimate_time_with_performance_model(payload):
    """
    Job for performance models
    """
    try:
        config_file = os.environ.get('JUICER_CONF')
        if config_file is None:
            result = {
                'status': 'ERROR',
                'message': 'You must inform the JUICER_CONF env variable'
            }
            return result

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        print('-' * 20)
        print(payload)
        print('-' * 20)
        # retrieves the model
        model = get_model_info(config, payload)
        platform = payload['platform']
        storage = model['storage']
        cluster = get_cluster_info(config, payload)

        if storage['type'] == 'HDFS':
            path = copy_model_files_to_local(model, storage)

            if platform == 'spark':
                cores = get_cores_info(config)
                result = _estimate_for_spark(cores, path, payload)
            elif platform == 'keras' or platform == 'scikit-learn':
                gpus_configuration = get_gpus_configuration(cluster)
                result = _estimate_for_keras(
                    gpus_configuration, path, payload, model)
            else:
                result = {
                    'status': 'ERROR',
                    'message': 'Unsupported platform: {}'.format(platform)
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


def _estimate_for_keras(gpus_configuration, path, payload, model):
    predictor = gpu_prediction.GpuPrediction(
        path, gpus_configuration)
    iterations = payload['iterations']
    pred_iterations = payload['data_size'] / payload['batch_size'] * iterations
    prediction = predictor.generate_predictions(
        str(os.path.basename(model['path']).split('.')[0]),
        payload.get('data_type', 'image'), payload['data_size'],
        payload['batch_size'], pred_iterations,
        payload['deadline'])
    print('predictor.generate_predictions(',
          str(os.path.basename(model['path']).split('.')[0]), ',',
          payload.get('data_type', 'image'), payload['data_size'],
          payload['batch_size'], payload['iterations'],
          payload['deadline'], ')')
    print('** Prediction **')
    print(prediction)
    results = [[list([[cores, time / 60.0] for cores, time in pair.items()])]
               for pair in prediction.values()][0]
    result = {
        'status': 'OK',
        'deadline': payload['deadline'],
        'result': results,
        'models': list(prediction.keys())
    }
    return result


def get_cluster_info(config, payload):
    stand_conf = config['juicer']['services']['stand']
    cluster = stand_service.get_cluster_info(
        stand_conf['url'], str(stand_conf['auth_token']),
        payload['cluster_id'])
    return cluster


def get_gpus_configuration(cluster):
    return {
        'M60': {1, 2, 3, 4}
    }


def _estimate_for_spark(cores, path, payload):
    print("Processando com ", int(payload.get('data_size')))
    optimizer = opt_ic.Optimizer(
        os.path.splitext(os.path.basename(path))[0],
        int(payload.get('data_size')), os.path.dirname(path))
    # Deadline is in seconds, model expects milliseconds
    deadline = int(payload.get('deadline', 3600)) * 60000
    result = {
        'status': 'OK',
        'deadline': deadline,
        'cores': cores,
        'result': [
            optimizer.solve(core, deadline)
            for core in cores
            ]
    }
    result['result'] = [v if v != float("inf") else -1
                        for v in result['result']]
    return result


def copy_model_files_to_local(model, storage):
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
        if model['path'].endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                # Directory has the same name as the model
                zip_ref.extractall(path[:-4])
            path = path[:-4]
    return path


def get_cores_info(cluster):
    cores = [2, 4, 8, 16]
    return cores


def get_model_info(config, payload):
    model_id = payload['model_id']
    limonero_conf = config['juicer']['services']['limonero']
    model = limonero_service.get_model_info(
        limonero_conf['url'], str(limonero_conf['auth_token']),
        model_id)
    return model


def cache_vallum_data(payload):
    try:
        config_file = os.environ.get('JUICER_CONF')
        if config_file is None:
            result = {
                'status': 'ERROR',
                'message': 'You must inform the JUICER_CONF env variable'
            }
            return result

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        print('-' * 20)
        job = get_current_job()
        print('Job Id: ', job.id)
        print(payload)
        print('-' * 20)
        total = perform_copy(config, payload.get('data_source_id'),
                             payload.get('storage_id'),
                             payload.get('path'))
        return {
            'status': 'OK',
            'message': '{} file(s) copied from Vallum to Storage.'.format(
                total),
        }
    except Exception as ex:
        result = {
            'status': 'ERROR',
            'message': 'Internal error: {}'.format(ex),
            'stack': traceback.format_exc()
        }
    return result
