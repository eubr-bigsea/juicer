# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent


from juicer.operation import Operation



class DataReaderOperation(Operation):


    HEADER_PARAM = 'header'
    SCHEMA = 'infer_schema' # FROM_VALUES, FROM_LIMONEIRO, NO
    NULL_VALUES_PARAM = 'null_values'
    SEPARATOR = 'separator'


    def __init__(self, parameters, named_inputs,  named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.name_file = parameters['data_source']
        self.separator = parameters.get(self.SEPARATOR, ',')
        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)
        self.schema = parameters.get(self.SCHEMA, "FROM_VALUES")


        self.null_values = [v.strip() for v in parameters.get(self.NULL_VALUES_PARAM, '').split(",")]

        self.has_code = len(named_outputs)>0
        if self.has_code:
            self.has_import = "from functions.data.ReadData import ReadParallelFile\n"



    def generate_code(self):

        code =  """
                    numFrag  = 4
                    {output} = ReadParallelFile('{input}','{separator}', {header},'{schema}',{null_values})
                """.format(output   = self.output,
                           input    = self.name_file,
                           separator= self.separator,
                           header   = self.header,
                           schema   = self.schema,
                           null_values = self.null_values)
        return dedent(code)


class SaveOperation(Operation):
    """
    Saves the content of the DataFrame at the specified path

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
    FORMAT_JSON = 'JSON'


    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = len(named_inputs) == 1
        if self.has_code:

            self.name   = parameters.get(self.NAME_PARAM)
            self.format = parameters.get(self.FORMAT_PARAM,'CSV')
            self.path   = parameters.get(self.PATH_PARAM)
            self.mode   = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
            self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)
            self.has_import = "from functions.data.SaveData import SaveOperation\n"

            self.output_tmp = "{}_tmp".format(self.named_inputs['input data'])

            if self.path == None:
                self.filename= self.name
            elif len(self.path)>0:
                self.filename= self.path+"/"+self.name
            else:
                self.filename= self.name



        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output_tmp

    def generate_code(self):

        code_save = ''
        if self.format == self.FORMAT_CSV:
            code_save = """
                        numFrag = 4
                        settings = dict()
                        settings['filename'] = '{output}'
                        settings['mode']     = '{mode}'
                        settings['header']   = {header}
                        settings['format']   = '{format}'
                        {tmp} = SaveOperation({input}, settings,numFrag)
                        """.format( tmp     = self.output_tmp,
                                    output  =self.filename,
                                    input   =self.named_inputs['input data'],
                                    mode    =self.mode,
                                    header  =self.header,
                                    format  =self.format )
             # Need to generate an output, even though it is not used.
             #code_save += '\n{0}_tmp = {0}'.format(self.named_inputs['input data'])

        elif self.format == self.FORMAT_PICKLE:
             pass


        code = dedent(code_save)

        return code




class WorkloadBalancerOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = (len(self.named_inputs) == 1) and (len(self.named_outputs) == 1)

        if self.has_code:
            self.has_import = "from functions.data.Balancer import WorkloadBalancerOperation\n"

    def generate_code(self):
        if self.has_code:
            code = """
                        numFrag = 4
                        {out} = WorkloadBalancerOperation({input},numFrag)
                    """.format( out     = self.named_outputs['output data'],
                                input   =self.named_inputs['input data'])

            return dedent(code)


class ChangeAttributesOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)
        self.has_code = (len(named_inputs) == 1) and (len(named_outputs)>0)

        if 'attributes' not in self.parameters:
            self.has_code = False
            msg = "Parameters '{}' must be informed for task {}"
            raise ValueError(msg.format('attributes', self.__class__.__name__))

        if self.has_code:
            self.has_import = "from functions.data.AttributesChanger import AttributesChangerOperation\n"

def generate_code(self):
    if self.has_code:

        new_name = self.parameters.get('new_name',[])
        new_data_type = self.parameters.get('new_data_type','keep')

        code = """
            numFrag  = 4
            settings = dict()
            settings['new_data_type'] = {new_data_type}
            settings['attributes']    = {att}
            settings['new_name']      = {new_name}
            {out} = AttributesChangerOperation({input},settings,numFrag)
            """.format(out   = self.named_outputs['output data'],
                       input = self.named_inputs['input data'],
                       new_name = new_name,
                       new_data_type = new_data_type,
                       att = self.parameters['attributes']
                       )
        return dedent(code)
    else:
        msg = "Parameters '{}' and '{}' must be informed for task {}"
        raise ValueError(msg.format('[]inputs',  '[]outputs', self.__class__))
