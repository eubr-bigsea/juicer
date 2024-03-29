# -*- coding: utf-8 -*-


import json
import logging

import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def get_platform(base_url, token, platform_id):
    return query_tahiti(base_url, '/platforms', token, platform_id)


def query_tahiti(base_url, item_path, token, item_id, qs=None):
    headers = {'X-Auth-Token': token}

    if item_id == '' or item_id is None:
        url = '{}/{}'.format(
            base_url, item_path if item_path[0] != '/' else item_path[1:])
    else:
        url = '{}/{}/{}'.format(
            base_url, item_path if item_path[0] != '/' else item_path[1:],
            item_id)
    if qs:
        url += '?' + qs
    log.debug(_('Querying Tahiti URL: %s'), url)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(_(
            "Error loading data from tahiti: id {}: HTTP {} - {}  ({})").format(
            item_id, r.status_code, r.text, url))


def save_workflow(base_url: str, token: str, workflow: str) -> int:
    url = f'{base_url}/workflows'
    headers = {
        'X-Auth-Token': str(token),
        'Content-type': 'application/json'
    }

    r = requests.post(url, data=workflow, headers=headers)
    if r.status_code == 200:
        return r.json().get('id')
    else:
        raise RuntimeError(_("Error saving workflow: {}").format(
            r.text))
