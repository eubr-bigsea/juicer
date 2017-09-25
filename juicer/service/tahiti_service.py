# -*- coding: utf-8 -*-
import json
import logging

import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def query_tahiti(base_url, item_path, token, item_id):
    headers = {'X-Auth-Token': token}

    if item_id == '':
        url = '{}/{}'.format(base_url, item_path)
    else:
        url = '{}/{}/{}'.format(base_url, item_path, item_id)

    log.debug(_('Querying Tahiti URL: %s'), url)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(_(
            u"Error loading storage id {}: HTTP {} - {}  ({})").format(
                item_id, r.status_code, r.text, url))


def get_storage_info(base_url, token, storage_id):
    return query_tahiti(base_url, 'storages', token, storage_id)


def get_data_source_info(base_url, token, data_source_id):
    return query_tahiti(base_url, '', token, data_source_id)
