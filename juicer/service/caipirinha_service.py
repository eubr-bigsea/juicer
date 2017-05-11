# -*- coding: utf-8 -*-
import json
import logging

import time

import happybase
import requests
import urlparse

from juicer.service import limonero_service
from juicer.util.dataframe_util import CustomEncoder

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def query_caipirinha(base_url, item_path, token, item_id, data):
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
            u"Error loading storage id {}: HTTP {} - {}".format(item_id,
                                                                r.status_code,
                                                                r.text))


def new_dashboard(config, title, user, workflow_id, workflow_name, job_id,
                  task_id, visualizations, emit_event_fn=None,
                  _type='VISUALIZATION'):
    data = dict(title=title, user=user, workflow_id=workflow_id,
                workflow_name=workflow_name, job_id=job_id, task_id=task_id,
                visualizations=visualizations)

    # Get Limonero configuration
    limonero_config = config['juicer']['services']['limonero']

    # Get Caipirinha configuration
    caipirinha_config = config['juicer']['services']['caipirinha']

    # Storage refers to the underlying environment used for storing
    # visualizations, e.g., Hbase
    storage = limonero_service.get_storage_info(
        limonero_config['url'], str(limonero_config['auth_token']),
        caipirinha_config['storage_id'])

    # Get hbase hostname and port
    parsed_url = urlparse.urlparse(storage['url'])
    connection = happybase.Connection(host=parsed_url.hostname,
                                      port=parsed_url.port)

    vis_table = connection.table('visualization')
    batch = vis_table.batch(timestamp=int(time.time()))

    if emit_event_fn is not None:
        emit_event_fn(
            'update task', status='RUNNING',
            identifier=task_id,
            task={'id': task_id},
            message='Saving visualizations',
            type='STATUS')

    # import pdb
    # pdb.set_trace()
    for visualization in visualizations:
        # HBase value composed by several columns, the last one refers
        # to the visualization data
        vis_value = {
            b'cf:user': json.dumps(user, indent=4),
            b'cf:workflow': json.dumps({'id': workflow_id}),
            b'cf:title': visualization['title'],
            b'cf:column_names': visualization['model'].column_names,
            b'cf:orientation': visualization['model'].orientation,
            b'cf:attributes': json.dumps({
                'id': visualization['model'].id_attribute,
                'value': visualization['model'].value_attribute
            }),
            b'cf:data': json.dumps(visualization['model'].get_data(),
                                   cls=CustomEncoder),
            b'cf:schema': visualization['model'].get_schema()
        }
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

        del visualization['model']
        batch.put(b'{job_id}-{task_id}'.format(task_id=visualization['task_id'],
                                               job_id=job_id),
                  vis_value)
    batch.send()

    # Ensure HBase connection is closed
    connection.close()

    r = query_caipirinha(caipirinha_config['url'], 'dashboards',
                         caipirinha_config['auth_token'], '',
                         json.dumps(data))
    if emit_event_fn is not None:
        emit_event_fn(
            'update task', status='COMPLETED',
            identifier=task_id,
            task={'id': task_id},
            message='Visualizations saved',
            type='STATUS')
    return r
