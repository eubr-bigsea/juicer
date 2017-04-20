# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent


from juicer.operation import Operation


#####from juicer.service import limonero_service
##from juicer.include.metadata import MetadataGet

class DataReader(Operation):


    HEADER_PARAM = 'header'
    SCHEMA = 'infer_schema' # FROM_VALUES, FROM_LIMONEIRO, NO

    def __init__(self, parameters, named_inputs,  named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.name_file = parameters['data_source']
        self.separator = parameters['separator']
        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)
        self.schema = parameters.get(self.SCHEMA, "FROM_VALUES")

        self.generate_code()

    def generate_code(self):

        code = " {} = ReadFromFile('{}','{}', {},'{}')"\
            .format(self.output,self.name_file, self.separator,self.header,self.schema)
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

    FORMAT_PICKLE = 'PICKLE'
    FORMAT_CSV = 'CSV'


    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name   = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM)
        self.path   = parameters.get(self.PATH_PARAM)
        self.mode   = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)





        if self.path != None:
            self.filename= self.path+"/"+self.name
        else:
            self.filename= self.name

        self.output = self.named_inputs['input data']

        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):

        code_save = ''
        if self.format == self.FORMAT_CSV:  #filename,data,mode,header

            code_save = """
                            tmp = SaveToFile('{output}',{input}, '{mode}',{header})
                        """.format( output=self.filename,
                                    input =self.named_inputs['input data'],
                                    mode  =self.mode,
                                    header=self.header)
             # Need to generate an output, even though it is not used.
             #code_save += '\n{0}_tmp = {0}'.format(self.named_inputs['input data'])

        elif self.format == self.FORMAT_PICKLE:
             pass


        code = dedent(code_save)

        return code
