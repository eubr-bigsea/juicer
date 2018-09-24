# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import logging

import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def query_tahiti(base_url, item_path, token, item_id, qs=None):
    headers = {'X-Auth-Token': token}

    if item_id == '':
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
            u"Error loading storage id {}: HTTP {} - {}  ({})").format(
            item_id, r.status_code, r.text, url))

