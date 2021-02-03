# -*- coding: utf-8 -*-

import json
import logging

import requests
from gettext import gettext

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

    if base_url.endswith('/'):
        base_url = base_url[:-1]

    if item_path.endswith('/'):
        item_path = item_path[:-1]

    if item_path:
        url = '{}/{}/{}'.format(base_url, item_path, item_id)
    else:
        url = '{}/{}'.format(base_url, item_id)

    # log.debug(gettext('Querying Limonero URL: %s'), url)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        log.error(gettext('Error querying Limonero URL: %s (%s: %s)'), url,
                  r.status_code, r.text)
        if r.status_code == 404:
            msg = gettext("not found")
        else:
            msg = r.text
        raise ValueError(gettext(
            "Error loading {} id {}: HTTP {} - {} ({})").format(
            item_path, item_id, r.status_code, msg, url))


def get_storage_info(base_url, token, storage_id):
    storage = query_limonero(base_url, 'storages', token, storage_id)
    return storage['data'][0]


def get_data_source_info(base_url, token, data_source_id):
    try:
        return query_limonero(base_url, 'datasources', token, data_source_id)
    except ValueError:
        raise ValueError(gettext('Data source not found'))

def get_model_info(base_url, token, model_id):
    try:
        return query_limonero(base_url, 'models', token, model_id)
    except ValueError:
        raise ValueError(gettext('Model not found'))

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
        log.error(gettext('Error saving model in Limonero URL: %s (%s: %s)'), url,
                  r.status_code, r.text)
        raise RuntimeError(gettext("Error saving model: {})").format(r.text))


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
        log.error(gettext('Error saving data source in Limonero URL: %s (%s: %s)'),
                  url, r.status_code, r.text)
        raise RuntimeError(gettext("Error saving datasource: {})").format(r.text))
def update_initialization_status(base_ur, payload, token):
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
        log.error(gettext('Error saving data source in Limonero URL: %s (%s: %s)'),
                  url, r.status_code, r.text)
        raise RuntimeError(gettext("Error saving datasource: {})").format(r.text))

