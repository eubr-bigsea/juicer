# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import logging

import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def save_job_source_code(base_url, token, job_id, source):
    headers = {
        'X-Auth-Token': str(token),
        'Content-Type': 'application/json'
    }

    url = '{}/jobs/{}/source-code'.format(base_url, job_id)

    r = requests.patch(url,
                       data=json.dumps({'secret': token, 'source': source}),
                       headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(
            u"Error loading data from stand: HTTP {} - {}  ({})".format(
                r.status_code, r.text, url))
