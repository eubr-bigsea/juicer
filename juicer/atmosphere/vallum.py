import json
import os
from urllib.parse import urlparse, parse_qs

import requests
import urllib3
import yaml
from juicer.service import limonero_service


def perform_copy(config, vallum_ds_id, target_id, path):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    services_config = config.get('juicer').get('services')
    limonero_config = services_config.get('limonero')

    limonero_url = limonero_config.get('url')
    token = str(limonero_config.get('auth_token'))
    vallum_ds = limonero_service.get_data_source_info(limonero_url, token,
                                                      vallum_ds_id)
    vallum_storage = vallum_ds.get('storage', {})
    if vallum_storage.get('type') != 'VALLUM':
        return {'status': 'ERROR', 'message': 'Storage is not VALLUM'}
    target_storage = limonero_service.get_storage_info(limonero_url, token,
                                                       target_id)
    if target_storage.get('type') != 'LOCAL':
        return {'status': 'ERROR',
                'message': 'Target storage must be of type LOCAL'}

    parsed = urlparse(vallum_storage.get('url'))
    base_url = '{}://{}:{}'.format(parsed.scheme, parsed.hostname,
                                   parsed.port or 80)
    url = base_url + parsed.path
    qs = parse_qs(parsed.query)
    database = qs.get('db', 'samples')[0]

    username = parsed.username
    password = parsed.password
    query = vallum_ds['command']
    mode = 'MN'
    thread = 1

    params = {
        "username": username,
        "password": password,
        "database": database,
        "mode": mode,
        "query": query,
        "thread": thread,
    }
    req = requests.post(url, params, verify=False)
    total = 0
    if req.status_code == 200:
        parsed_local = urlparse(target_storage.get('url'))
        target_dir = parsed_local.path + path  # '/vallum' + str(vallum_ds_id)
        obj = json.loads(req.text)
        for result in obj.get('result'):
            files = result.get('files')
            if files:
                uri_files = [base_url + urlparse(f.get('uri')).path for f in
                             files]
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                for vallum_file in uri_files:
                    file_req = requests.get(vallum_file, params, verify=False)
                    if file_req.status_code == 200:
                        final_filename = target_dir + '/' + \
                                         vallum_file.split('/')[-1]
                        print(final_filename)
                        total += 1
                        with open(final_filename, 'wb') as fout:
                            fout.write(file_req.content)
                    else:
                        raise ValueError('HTTP Status ' + file_req.status_code)
        return total
    else:
        raise ValueError('HTTP Status ' + req.status_code)


if __name__ == '__main__':
    config_filename = os.environ.get('JUICER_CONFIG')
    if config_filename is None:
        print('Inform JUICER_CONFIG')
        exit(1)
    with open(config_filename) as config_file:
        juicer_config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    perform_copy(juicer_config, 521, 7)
