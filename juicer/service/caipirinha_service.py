# -*- coding: utf-8 -*-
import json
import logging
import time
import urlparse

import happybase
import requests
from juicer.service import limonero_service
from juicer.util.dataframe_util import CustomEncoder

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def _update_caipirinha(base_url, item_path, token, item_id, data):
    headers = {'X-Auth-Token': token}

    if item_id == '':
        url = '{}/{}'.format(base_url, item_path)
    else:
        url = '{}/{}/{}'.format(base_url, item_path, item_id)

    log.debug('Querying Caipirinha URL: %s', url)

    r = requests.post(url, headers=headers, data=data)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(
            u"Error in URL {}: HTTP {} - {}".format(
                item_id, url, r.status_code, r.text))


def _emit_saving_visualization(emit_event_fn, task_id):
    if emit_event_fn is not None:
        emit_event_fn(
            'update task', status='RUNNING',
            identifier=task_id,
            task={'id': task_id},
            message='Saving visualizations',
            type='STATUS')


def _emit_saved_visualization(_type, emit_event_fn, visualization):
    if emit_event_fn is not None:
        emit_event_fn(
            'task result', status='COMPLETED',
            identifier=visualization['task_id'],
            task={'id': visualization['task_id']},
            message='Result generated',
            type=_type,
            title=visualization['model'].title,
            operation={'id': visualization['model'].type_id},
            operation_id=visualization['model'].type_id)


def _emit_completed(emit_event_fn, task_id):
    if emit_event_fn is not None:
        emit_event_fn(
            'update task', status='COMPLETED',
            identifier=task_id,
            task={'id': task_id},
            message='Visualizations saved',
            type='STATUS')


def _get_hbsase_visualization_format(user, visualization, workflow_id):
    return
    # vis_value = {
    #     b'cf:user': json.dumps(user, indent=4),
    #     b'cf:workflow': json.dumps({'id': workflow_id}),
    #     b'cf:title': visualization['title'],
    #     b'cf:column_names': visualization['model'].get_column_names(),
    #     b'cf:orientation': visualization['model'].orientation,
    #     b'cf:attributes': json.dumps({
    #         'id': visualization['model'].id_attribute,
    #         'value': visualization['model'].value_attribute
    #     }),
    #     b'cf:data': json.dumps(visualization['model'].get_data(),
    #                            cls=CustomEncoder),
    #     b'cf:schema': visualization['model'].get_schema()
    # }
    # return vis_value


def _get_params(config):
    # Get Limonero configuration
    limonero_config = config['juicer']['services']['limonero']
    # Get Caipirinha configuration
    caipirinha_config = config['juicer']['services']['caipirinha']
    # Storage refers to the underlying environment used for storing
    # visualizations, e.g., HBase
    storage = limonero_service.get_storage_info(
        limonero_config['url'], str(limonero_config['auth_token']),
        caipirinha_config['storage_id'])
    # Get HBase hostname and port
    parsed_url = urlparse.urlparse(storage['url'])
    connection = happybase.Connection(host=parsed_url.hostname,
                                      port=parsed_url.port)
    vis_table = connection.table('visualization')
    batch = vis_table.batch(timestamp=int(time.time()))
    return batch, caipirinha_config, connection


def new_dashboard(config, title, user, workflow_id, workflow_name, job_id,
                  task_id, visualizations, emit_event_fn=None,
                  _type='VISUALIZATION'):
    data = dict(title=title, user=user, workflow_id=workflow_id,
                workflow_name=workflow_name, job_id=job_id, task_id=task_id,
                visualizations=visualizations)

    batch, caipirinha_config, connection = _get_params(config)
    _emit_saving_visualization(emit_event_fn, task_id)

    for visualization in visualizations:
        # HBase value composed by several columns, the last one refers
        # to the visualization data
        # _get_hbsase_visualization_format(user, visualization,
        #                                  workflow_id)
        # vis_value = _get_hbsase_visualization_format(
        #     user, visualization, workflow_id)
        _emit_saved_visualization(_type, emit_event_fn, visualization)

        del visualization['model']
    #     batch.put(b'{job_id}-{task_id}'.format(task_id=visualization['task_id'],
    #                                            job_id=job_id), vis_value)
    # batch.send()

    # Ensure HBase connection is closed
    connection.close()

    r = _update_caipirinha(caipirinha_config['url'], 'dashboards',
                           caipirinha_config['auth_token'], '',
                           json.dumps(data))
    _emit_completed(emit_event_fn, task_id)
    return r


def new_visualization(config, user, workflow_id, job_id,
                      task_id, visualization, emit_event_fn=None,
                      _type='VISUALIZATION'):
    batch, caipirinha_config, connection = _get_params(config)

    _emit_saving_visualization(emit_event_fn, task_id)
    # vis_value = _get_hbsase_visualization_format(
    #     user, visualization, workflow_id)
    _emit_saved_visualization(_type, emit_event_fn, visualization)

    del visualization['model']
    # batch.put(b'{job_id}-{task_id}'.format(
    #     task_id=visualization['task_id'], job_id=job_id), vis_value)
    # batch.send()

    # Ensure HBase connection is closed
    connection.close()

    r = _update_caipirinha(
        caipirinha_config['url'], 'visualizations',
        caipirinha_config['auth_token'], '', json.dumps(visualization))

    _emit_completed(emit_event_fn, task_id)
    return r
