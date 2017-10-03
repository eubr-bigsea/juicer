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

    if item_path:
        url = '{}/{}/{}'.format(base_url, item_path, item_id)
    else:
        url = '{}/{}'.format(base_url, item_id)

    log.debug(_('Querying Limonero URL: %s'), url)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        log.error(_('Error querying Limonero URL: %s (%s: %s)'), url,
                  r.status_code, r.text)
        raise RuntimeError(_(
            u"Error loading {} id {}: HTTP {} - {}").format(item_path, item_id,
                                                            r.status_code,
                                                            r.text))


def get_storage_info(base_url, token, storage_id):
    return query_limonero(base_url, 'storages', token, storage_id)


def get_data_source_info(base_url, token, data_source_id):
    return query_limonero(base_url, '', token, data_source_id)


def register_model(base_url, payload, token):
    url = "{}/models".format(remove_initial_final_path_separator(base_url))

    headers = {
        'x-auth-token': token,
        'content-type': "application/json",
        'cache-control': "no-cache"
    }
    r = requests.request("POST", url, data=json.dumps(payload), headers=headers)

    if r.status_code == 200:
        return json.loads(r.text)
    else:
        log.error(_('Error saving model in Limonero URL: %s (%s: %s)'), url,
                  r.status_code, r.text)
        raise RuntimeError(_("Error saving model: {})").format(r.text))


def register_datasource(base_url, payload, token, mode=''):
    url = "{url}/datasources?mode={mode}".format(
        url=remove_initial_final_path_separator(base_url), mode=mode)

    headers = {
        'x-auth-token': token,
        'content-type': "application/json",
        'cache-control': "no-cache"
    }
    r = requests.request("POST", url, data=json.dumps(payload), headers=headers)

    if r.status_code == 200:
        return json.loads(r.text)
    else:
        log.error(_('Error saving data source in Limonero URL: %s (%s: %s)'),
                  url, r.status_code, r.text)
        raise RuntimeError(_("Error saving datasource: {})").format(r.text))
