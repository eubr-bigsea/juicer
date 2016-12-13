# -*- coding: utf-8 -*-
import ast
import json
import pprint
from textwrap import dedent

from juicer.dist.metadata import MetadataGet
from juicer.service import limonero_service
from juicer.spark.operation import Operation


class DataReader(Operation):
    """
    Reads a database.
    Parameters:
    - Limonero database ID
    """
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    INFER_SCHEMA_PARAM = 'infer_schema'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_DATA'
    DO_NOT_INFER = 'NO'

    LIMONERO_TO_SPARK_DATA_TYPES = {
        "INTEGER": 'IntegerType',
        "TEXT": 'StringType',
        "LONG": 'LongType',
        "DOUBLE": 'DoubleType',
        "DATETIME": 'TimestampType',
    }

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
            self.header = bool(parameters.get(self.HEADER_PARAM, False))
            self.sep = parameters.get(self.SEPARATOR_PARAM, ',')
            self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                               self.INFER_FROM_LIMONERO)

            metadata_obj = MetadataGet('123456')
            self.metadata = metadata_obj.get_metadata(self.database_id)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__))

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero?
        # such as header and sep.
        # print "\n\n",self.metadata,"\n\n"
        code = []
        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        if len(self.outputs) == 1:
            output = self.outputs[0]
            if infer_from_limonero:
                if 'attributes' in self.metadata:
                    code.append('schema_{0} = StructType()'.format(output))
                    for attr in self.metadata.get('attributes', []):
                        data_type = self.LIMONERO_TO_SPARK_DATA_TYPES[
                            attr['type']]

                        # Notice: According to Spark documentation, nullable
                        # option of StructField is just a hint and when loading
                        # CSV file, it won't work. So, we are adding this
                        # information in metadata.

                        metadata = {k: attr[k] for k in
                                    ['feature', 'label', 'nullable', 'type',
                                     'size', 'precision', 'enumeration',
                                     'missing_representation'] if attr[k]}
                        code.append("schema_{0}.add('{1}', {2}(), {3}, {4})"
                                    .format(output, attr['name'], data_type,
                                            attr['nullable'],
                                            pprint.pformat(metadata, indent=0)))
                    code.append("")
                else:
                    raise ValueError(
                        "Metadata do not include attributes information")
            else:
                code.append('schema_{0} = None'.format(output))
            if self.metadata['format'] == 'CSV':
                code.append(
                    "url_{0} = '{1}'".format(output, self.metadata['url']))
                code.append(
                    "{0} = spark_session.read.csv(url_{0}, schema=schema_{0}, "
                    "header={1}, sep='{2}', inferSchema={3})".format(
                        output, self.header, self.sep,
                        infer_from_data))

                # FIXME: Evaluate if it is good idea to always use cache
                code.append('{}.cache()'.format(output))

            elif self.metadata['format'] == 'PARQUET_FILE':
                # TO DO
                pass
            elif self.metadata['format'] == 'JSON_FILE':
                # TO DO
                pass

        return '\n'.join(code)


class Save(Operation):
    """
    Saves the content of the DataFrame at the specified path
    and generate the code to call the Limonero API.
    Parameters:
        - Database name
        - Path for storage
        - Storage ID
        - Database tags
        - Workflow that generated the database
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_ID_PARAM = 'storage'
    FORMAT_PARAM = 'format'
    TAGS_PARAM = 'tags'
    OVERWRITE_MODE_PARAM = 'mode'
    HEADER_PARAM = 'header'

    MODE_ERROR = 'error'
    MODE_APPEND = 'append'
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'

    FORMAT_PARQUET = 'PARQUET'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'
    WORKFLOW_JSON_PARAM = 'workflow_json'
    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM)
        self.url = parameters.get(self.PATH_PARAM)
        self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        self.tags = ast.literal_eval(parameters.get(self.TAGS_PARAM, '[]'))
        self.path = parameters.get(self.PATH_PARAM)

        self.workflow_json = parameters.get(self.WORKFLOW_JSON_PARAM, '')

        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        self.header = parameters.get(self.HEADER_PARAM, True) in (1, '1', True)

        self.user = parameters.get(self.USER_PARAM)
        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)
        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        # Retrieve Storage URL
        # @FIXME Hardcoded!
        storage = limonero_service.get_storage_info(
            'http://beta.ctweb.inweb.org.br/limonero', '123456',
            self.storage_id)

        final_url = '{}{}{}'.format(storage['url'], self.path,
                                    self.name.replace(' ', '_'))

        code_save = ''
        if self.format == self.FORMAT_CSV:
            code_save = dedent("""
            {}.write.csv('{}',
                         header={}, mode='{}')""".format(
                self.inputs[0], final_url, self.header, self.mode))
            # Need to generate an output, even though it is not used.
            code_save += '\n{0}_tmp = {0}'.format(self.inputs[0])
        elif self.format == self.FORMAT_PARQUET:
            code_save = dedent("""
            {}.write.parquet('{}', mode='{}')""".format(self.inputs[0],
                                                        final_url, self.mode))
            # Need to generate an output, even though it is not used.
            code_save += '\n{0}_tmp = {0}'.format(self.inputs[0])
        elif self.format == self.FORMAT_JSON:
            pass

        code = dedent(code_save)

        if not self.workflow_json == '':
            code_api = """
                # Code to update Limonero metadata information
                from metadata import MetadataPost
                types_names = {{
                'IntegerType': "INTEGER",
                'StringType': "TEXT",
                'LongType': "LONG",
                'DoubleType': "DOUBLE",
                'TimestampType': "DATETIME",
                }}
                schema = []
                # nullable information is also stored in metadata
                # because Spark ignores this information when loading CSV files
                for att in {0}.schema:
                    schema.append({{
                      'name': att.name,
                      'dataType': types_names[str(att.dataType)],
                      'nullable': att.nullable or attr.metadata.get('nullable'),
                      'metadata': att.metadata,
                    }})
                parameters = {{
                    'name': "{1}",
                    'format': "{2}",
                    'storage_id': {3},
                    'provenience': '{4}',
                    'description': "{5}",
                    'user_id': "{6}",
                    'user_login': "{7}",
                    'user_name': "{8}",
                    'workflow_id': "{9}",
                    'url': "{10}",
                }}
                instance = MetadataPost('{11}', schema, parameters)
                """.format(self.inputs[0], self.name, self.format,
                           self.storage_id,
                           self.workflow_json,
                           self.user['name'],
                           self.user['id'],
                           self.user['login'],
                           self.user['name'],
                           self.workflow_id, final_url, "123456"
                           )
            code += dedent(code_api)

        return code


class ReadCSV(Operation):
    """
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.url = parameters['url']
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = "True"
        try:
            self.separator = parameters['separator']
        except KeyError:
            self.separator = ";"

    def generate_code(self):
        code = """{} = spark_session.read.csv('{}',
            header={}, sep='{}' ,inferSchema=True)""".format(
            self.outputs[0], self.url, self.header, self.separator)
        return dedent(code)
