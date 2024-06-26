# -*- coding: utf-8 -*-


import json
import logging

import re
import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def save_job_source_code(base_url, token, job_id, source):
    headers = {
        'X-Auth-Token': str(token),
        'Content-Type': 'application/json'
    }
    handle_protected = re.compile(r'(.+?)\s*=\s*(.+?)\s*#protect (.+)')
    final_source = handle_protected.sub(r'\1 = \3', source)
    url = '{}/jobs/{}/source-code'.format(base_url, job_id)

    r = requests.patch(url,
                       data=json.dumps({'secret': token, 'source': final_source},
                                       sort_keys=True),
                       headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        log.warn("Error saving source code in stand: HTTP %s %s  (%s)",
                r.status_code, r.text, url)
        return {}


def get_cluster_info(base_url, token, cluster_id):
    headers = {
        'X-Auth-Token': str(token),
        'Content-Type': 'application/json'
    }

    url = '{}/clusters/{}'.format(base_url, cluster_id)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(
            "Error loading data from stand: HTTP {} - {}  ({})".format(
                r.status_code, r.text, url))
