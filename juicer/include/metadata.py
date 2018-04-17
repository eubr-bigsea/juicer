import json

import requests


class MetadataGet:
    def __init__(self, url, token):
        self.url = '{}/datasources/'.format(url)
        self.token = token

    def get_metadata(self, _id):
        # Call the Limonero API to get the metadata
        link = self.url + '/{}?token={}'.format(str(_id),
                                                str(self.token)).replace('//',
                                                                         '/')
        data = requests.get(link)
        return json.loads(data.text)


class MetadataPost:
    def __init__(self, url, token, df_schema, parameters):
        self.url = '{}/datasources/'.format(url)
        self.token = token
        self.payload = None
        self.headers = None
        self.querystring = None
        self.df_schema = df_schema
        self.parameters = parameters

        self.build_querystring()
        self.build_headers()
        self.build_payload()
        self.post_metadata()

    def post_metadata(self):
        response = requests.post(self.url, data=json.dumps(self.payload),
                                 headers=self.headers, params=self.querystring)
        if response.status_code != 200:
            print _(
                "\n ERROR! Status code: {},{} \n".format(response.status_code,
                                                         response.text))

    def build_querystring(self):
        self.querystring = {"token": "{0}".format(self.token)}

    def build_headers(self):
        self.headers = {
            'x-auth-token': "{0}".format(self.token),
            'content-type': "application/json",
            'cache-control': "no-cache",
        }

    def build_payload(self):
        attributes = []
        for schema in self.df_schema:
            attribute = {'enumeration': "False"}
            if schema['metadata'].has_key('feature'):
                attribute['feature'] = schema['metadata']['feature']
            else:
                attribute['feature'] = "True"
            if schema['metadata'].has_key('label'):
                attribute['label'] = schema['metadata']['label']
            else:
                attribute['label'] = "True"
            attribute['name'] = schema['name']
            attribute['nullable'] = schema['nullable']
            attribute['type'] = schema['dataType']
            attributes.append(attribute)

        self.payload = {
            "attributes": attributes,
            "enabled": True,
            "url": self.parameters['url'],
            "read_only": True,
            "name": self.parameters['name'],
            "format": self.parameters['format'],
            "provenience": self.parameters['provenience'],
            "storage_id": self.parameters['storage_id'],
            "description": self.parameters['description'],
            "user_id": self.parameters['user_id'],
            "user_login": self.parameters['user_login'],
            "user_name": self.parameters['user_name'],
            "workflow_id": self.parameters['workflow_id'],
        }
