# -*- coding: utf-8 -*-
import ast
import pprint
from textwrap import dedent

from juicer.include.metadata import MetadataGet
from juicer.operation import Operation
from juicer.service import limonero_service


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
    NULL_VALUES_PARAM = 'null_values'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_VALUES'
    DO_NOT_INFER = 'NO'

    LIMONERO_TO_SPARK_DATA_TYPES = {
        "INTEGER": 'types.IntegerType',
        "TEXT": 'types.StringType',
        "LONG": 'types.LongType',
        "FLOAT": 'types.FloatType',
        "DOUBLE": 'types.DoubleType',
        "DATETIME": 'types.TimestampType',
        "CHARACTER": 'types.StringType',
    }
    SEPARATORS = {
        '{tab}': '\\t'
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = len(self.named_outputs) > 0 or True
        if self.has_code:
            if self.DATA_SOURCE_ID_PARAM in parameters:
                self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
                self.header = parameters.get(
                    self.HEADER_PARAM, False) not in ('0', 0, 'false', False)
                self.sep = parameters.get(self.SEPARATOR_PARAM, ',')
                if self.sep in self.SEPARATORS:
                    self.sep = self.SEPARATORS[self.sep]
                self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                                   self.INFER_FROM_LIMONERO)
                self.null_values = [v.strip() for v in parameters.get(
                    self.NULL_VALUES_PARAM, '').split(",")]

                metadata_obj = MetadataGet('123456')

                # @FIXME Parameter
                url = 'http://beta.ctweb.inweb.org.br/limonero/datasources'
                token = '123456'
                self.metadata = limonero_service.get_data_source_info(
                    url, token, self.database_id)
            else:
                raise ValueError(
                    "Parameter '{}' must be informed for task {}".format(
                        self.DATA_SOURCE_ID_PARAM, self.__class__))
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero?
        # such as header and sep.
        # print "\n\n",self.metadata,"\n\n"
        code = []
        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        if self.has_code:
            if infer_from_limonero:
                if 'attributes' in self.metadata:
                    code.append(
                        'schema_{0} = types.StructType()'.format(self.output))
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
                        code.append("schema_{0}.add('{1}', {2}(), {3},\n{5}{4})"
                                    .format(self.output, attr['name'],
                                            data_type,
                                            attr['nullable'],
                                            pprint.pformat(metadata, indent=0),
                                            ' ' * 20
                                            ))
                    code.append("")
                else:
                    raise ValueError(
                        "Metadata do not include attributes information")
            else:
                code.append('schema_{0} = None'.format(self.output))

            # import pdb
            # pdb.set_trace()

            if self.metadata['format'] in ['CSV', 'TEXT']:
                code.append("url = '{url}'".format(url=self.metadata['url']))
                null_option = ''.join(
                    [".option('nullValue', '{}')".format(n) for n in
                     self.null_values]) if self.null_values else ""

                if self.metadata['format'] == 'CSV':
                    code_csv = dedent("""
                        {0} = spark_session.read{4}\\
                            .option('treatEmptyValuesAsNulls', 'true')\\
                            .csv(url, schema=schema_{0},
                                header={1}, sep='{2}', inferSchema={3},
                                mode='DROPMALFORMED')""".format(
                        self.output, self.header, self.sep, infer_from_data,
                        null_option))
                    code.append(code_csv)
                else:
                    code_csv = """{output} = spark_session.read\\
                           {null_option}\\
                           .option('treatEmptyValuesAsNulls', 'true')\\
                           .text(url)""".format(output=self.output,
                                                null_option=null_option)
                    code.append(code_csv)
                # FIXME: Evaluate if it is good idea to always use cache
                code.append('{}.cache()'.format(self.output))

            elif self.metadata['format'] == 'PARQUET_FILE':
                # TO DO
                pass
            elif self.metadata['format'] == 'JSON_FILE':
                # TO DO
                pass
            elif self.metadata['format'] == 'LIB_SVM':
                # Test
                format_libsvm = 'libsvm'
                code.append(
                    "url_{0} = '{1}'".format(self.output, self.metadata['url']))

                code_csv = """{0} = spark_session.read\\
                            .format('{2}')\\
                            .load(url_{0}, mode='DROPMALFORMED')
                """.format(self.output,
                           infer_from_data,
                           format_libsvm)

                code.append(code_csv)
                # # FIXME: Evaluate if it is good idea to always use cache
                code.append('{}.cache()'.format(self.output))

        return '\n'.join(code)

    def get_output_names(self, sep=", "):
        return self.output

    def get_data_out_names(self, sep=','):
        return self.output


class SaveOperations(Operation):
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

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

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
        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        # Retrieve Storage URL
        # @FIXME Hardcoded!
        storage = limonero_service.get_storage_info(
            'http://beta.ctweb.inweb.org.br/limonero', '123456',
            self.storage_id)

        final_url = '{}/{}/{}'.format(storage['url'], self.path,
                                      self.name.replace(' ', '_'))
        code_save = ''
        if self.format == self.FORMAT_CSV:
            code_save = dedent("""
            {}.write.csv('{}',
                         header={}, mode='{}')""".format(
                self.named_inputs['input data'], final_url, self.header,
                self.mode))
            # Need to generate an output, even though it is not used.
        elif self.format == self.FORMAT_PARQUET:
            code_save = dedent("""
            {}.write.parquet('{}', mode='{}')""".format(
                self.named_inputs['input data'],
                final_url, self.mode))
            # Need to generate an output, even though it is not used.
            code_save += '\n{0}_tmp = {0}'.format(
                self.named_inputs['input data'])
        elif self.format == self.FORMAT_JSON:
            pass

        code = dedent(code_save)

        if not self.workflow_json == '':
            code_api = """
                # Code to update Limonero metadata information
                from juicer.dist.metadata import MetadataPost
                types_names = {{
                'IntegerType': "INTEGER",
                'types.StringType': "TEXT",
                'LongType': "LONG",
                'DoubleType': "DOUBLE",
                'TimestampType': "DATETIME",
                'FloatType': "FLOAT"
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
                """.format(self.named_inputs['input data'], self.name,
                           self.format,
                           self.storage_id,
                           self.workflow_json,
                           self.user['name'],
                           self.user['id'],
                           self.user['login'],
                           self.user['name'],
                           self.workflow_id, final_url, "123456"
                           )
            code += dedent(code_api)
            # No return
            code += '{} = None'.format(self.output)

        return code


class ReadCSV(Operation):
    """
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
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
            header={}, sep='{}',inferSchema=True)""".format(
            self.outputs[0], self.url, self.header, self.separator)
        return dedent(code)


class ChangeAttribute(Operation):
    ATTRIBUTES_PARAM = 'attributes'
    IS_FEATURE_PARAM = 'is_feature'
    IS_LABEL_PARAM = 'is_label'
    NULLABLE_PARAM = 'nullable'
    NEW_NAME_PARAM = 'new_name'
    NEW_DATA_TYPE_PARAM = 'new_data_type'
    KEEP_VALUE = 'keep'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        del self.parameters['workflow_json']
        code = []
        if self.parameters.get(self.NEW_DATA_TYPE_PARAM,
                               'keep') == self.KEEP_VALUE:
            # Do not require processing data frame, change only meta data
            code.append('{0} = {1}'.format(self.output, self.inputs[0]))

            for attr in self.attributes:
                code.append(
                    "\ninx_{0} = [i for i, _ in enumerate({0}.schema) "
                    "if _.name.lower() == '{1}']".format(self.output,
                                                         attr.lower()))

                nullable = self.parameters.get(self.NULLABLE_PARAM,
                                               self.KEEP_VALUE)
                if nullable != self.KEEP_VALUE:
                    code.append(
                        ChangeAttribute.change_meta(
                            self.output, attr, 'nullable', nullable == 'true'))

                feature = self.parameters.get(self.IS_FEATURE_PARAM,
                                              self.KEEP_VALUE)
                if feature != self.KEEP_VALUE:
                    code.append(
                        ChangeAttribute.change_meta(
                            self.output, attr, 'feature', feature == 'true'))

                label = self.parameters.get(self.IS_LABEL_PARAM,
                                            self.KEEP_VALUE)
                if label != self.KEEP_VALUE:
                    code.append(
                        ChangeAttribute.change_meta(
                            self.output, attr, 'label', label == 'true'))

            format_name = self.parameters[self.NEW_NAME_PARAM]
            if format_name:
                rename = [
                    "withColumnRenamed('{}', '{}')".format(
                        attr, ChangeAttribute.new_name(format_name, attr)) for
                    attr in self.attributes]

                code.append('{0} = {0}.{1}'.format(
                    self.output, '\\\n              .'.join(rename)))

        else:
            # Changing data type requires to rebuild data frame
            pass
        return '\n'.join(code)  # json.dumps(self.parameters)

    @staticmethod
    def new_name(format_new_name, name):
        return format_new_name.format(name)

    @staticmethod
    def change_meta(output, attr_name, meta_name, value):
        return dedent("if inx_{0}:\n"
                      "    {0}.schema.fields[inx_{0}[0]]"
                      ".metadata['{2}'] = {3}".format(output, attr_name,
                                                      meta_name, value))


class ExternalInputOperation(Operation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.has_code = len(self.output) > 0

    def generate_code(self):
        code = """{out} = None""".format(out=self.output)
        return code
