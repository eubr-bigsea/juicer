# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent

from juicer.include.metadata import MetadataGet
from juicer.operation import Operation
from juicer.service import limonero_service


class DataReader(Operation):
    def __init__(self, parameters, named_inputs,  named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.name_file = parameters['data_source']
        self.separator = parameters['separator']
        #if  self.separator
        self.generate_code()

    def generate_code(self):

        code = " {} = ReadFromFile({},'{}', {})"\
            .format(self.output,self.name_file, self.separator,None) # TO DO: Change None !
        return dedent(code)


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

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM)
        # self.url = parameters.get(self.PATH_PARAM)
        # self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        # self.tags = ast.literal_eval(parameters.get(self.TAGS_PARAM, '[]'))
        # self.path = parameters.get(self.PATH_PARAM)
        #
        # self.workflow_json = parameters.get(self.WORKFLOW_JSON_PARAM, '')
        #
        # self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        # self.header = parameters.get(self.HEADER_PARAM, True) in (1, '1', True)
        #
        # self.user = parameters.get(self.USER_PARAM)
        # self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)
        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):

        code_save = ''
        if self.format == self.FORMAT_CSV:
             self.outputs = "tmp"
             code_save = dedent("""
                                   tmp = SaveToFile('{output}',{input}, ',')
                                """.format(output=self.name, input=self.named_inputs['input data']))
            # Need to generate an output, even though it is not used.
             #code_save += '\n{0}_tmp = {0}'.format(self.inputs[0])

        elif self.format == self.FORMAT_JSON:
             pass

        code = dedent(code_save)

        return code
