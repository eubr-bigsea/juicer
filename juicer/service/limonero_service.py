# -*- coding: utf-8 -*-
import json
import logging

import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def remove_initial_final_path_separator(path):
    if path.endswith('/'):
        path = path[:-1]
    if path.startswith('/'):
        path = path[1:]
    return path


def query_limonero(base_url, item_path, token, item_id):
    headers = {'X-Auth-Token': token}

    base_url = remove_initial_final_path_separator(base_url)
    item_path = remove_initial_final_path_separator(item_path)
    item_id = remove_initial_final_path_separator(str(item_id))

    url = '{}/{}/{}'.format(base_url, item_path, item_id)

    log.debug('Querying Limonero URL: %s', url)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        log.debug('Error querying Limonero URL: %s (%s: %s)', url,
                  r.status_code, r.text)
        raise RuntimeError(
            u"Error loading {} id {}: HTTP {} - {}".format(item_path, item_id,
                                                           r.status_code,
                                                           r.text))


def get_storage_info(base_url, token, storage_id):
    return query_limonero(base_url, 'storages', token, storage_id)


def get_data_source_info(base_url, token, data_source_id):
    return query_limonero(base_url, '', token, data_source_id)
