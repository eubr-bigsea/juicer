import requests
import json

class MetadataGet:

    def __init__(self, token):
        self.url = "http://beta.ctweb.inweb.org.br/limonero/datasources"
        self.token = token

    def get_metadata(self, id):
        # Call the Limonero API to get the metadata
        link =  self.url + '/{}?token={}'.format(str(id), str(self.token))
        data = requests.get(link)
        return json.loads(data.text)


class MetadataPost:

    def __init__(self, token, df_schema, parameters):
        self.url = "http://beta.ctweb.inweb.org.br/limonero/datasources"
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
            #requests.post(self.url, data=self.payload,headers=self.headers, params=self.querystring)

            print "requests.post('{0}',data=json.dumps({1}),headers={2},params={3})".format(
                self.url, self.payload, self.headers, self.querystring)


    def build_querystring(self):
        self.querystring = {"token":"{0}".format(self.token)}



    def build_headers(self):
        self.headers = {
            'x-auth-token':"{0}".format(self.token),
            'content-type':"application/json",
            'cache-control':"no-cache",
        }


    def build_payload(self):
        attributes = []
        for schema in self.df_schema:
            attribute = {}
            attribute['enumeration'] = "False"
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
            "url": "hdfs://localhost:9000/test/data2",
            "read_only": True,
            "name": self.parameters['name'],
            "format": self.parameters['format'],
            "provenience": self.parameters['provenience'],
            "storage_id":self.parameters['storage_id'],
            "description":self.parameters['description'],
            "user_id":self.parameters['user_id'],
            "user_login":self.parameters['user_login'],
            "user_name":self.parameters['user_name'],
            "workflow_id":self.parameters['workflow_id'],
        }
