# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation
from juicer.compss.expression import Expression

class AddColumnsOperation(Operation): # ok
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)
        self.has_code = (len(named_inputs) == 2) and (len(named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.AddColumns import AddColumnsOperation\n"

    def generate_code(self):
        if self.has_code:

            code = """
            numFrag  = 4
            balanced = False #Currently, all data are considered unbalanced
            {0} = AddColumnsOperation({1},{2},balanced,numFrag)
            """.format(self.named_outputs['output data'],
                       self.named_inputs['input data 1'],
                       self.named_inputs['input data 2'])
            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs',  '[]outputs', self.__class__))

class AggregationOperation(Operation):
    """
    Computes aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum,
        count.
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)
        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Aggregation import AggregationOperation\n"
        self.input_columns = parameters['attributes']
        self.input_aliases = {}
        self.input_operations = {}
        for dict in parameters['function']:
            att = dict['attribute']
            if att in self.input_operations:
                self.input_operations[att].append(dict['f'])
                self.input_aliases[att].append(dict['alias'])
            else:
                self.input_operations[att] = [dict['f']]
                self.input_aliases[att] = [dict['alias']]

    def generate_code(self):
        if self.has_code:

            code = """
            numFrag = 4
            settings = dict()
            settings['columns'] = {columns}
            settings['operation'] = {operations}
            settings['aliases']   = {aliases}
            {output} = AggregationOperation({input},settings,numFrag)
            """.format(output = self.named_outputs['output data'],
                       input  = self.named_inputs['input data'],
                       columns= self.input_columns,
                       aliases= self.input_aliases,
                       operations= self.input_operations)
            return dedent(code)


class CleanMissingOperation(Operation):
    """
    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values. Possible values include
          * "VALUE": replace by parameter "value",
          * "MEDIAN": replace by median value
          * "MODE": replace by mode value
          * "MEAN": replace by mean value
          * "REMOVE_ROW": remove entire row
          * "REMOVE_COLUMN": remove entire column
        - value: optional, used to replace missing values
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)


        self.attributes_CM    = self.parameters['attributes']
        self.mode_CM          = self.parameters.get('cleaning_mode',"REMOVE_ROW")
        self.value_CM         = self.parameters.get('value', None)
        self.has_code = all([
                any([self.value_CM is not None,  self.mode_CM != "VALUE"]),
                len(self.named_inputs) > 0,
                len(self.named_outputs) > 0
        ])

        if self.has_code:
            self.has_import = "from functions.etl.CleanMissing import CleanMissingOperation\n"


    def generate_code(self):
        if self.has_code:

            code = """
                numFrag = 4
                settings = dict()
                settings['attributes'] = {attributes}
                settings['cleaning_mode'] = '{cleaning_mode}'
                """.format(attributes = self.attributes_CM,
                           cleaning_mode = self.mode_CM)
            if self.mode_CM == "VALUE":
                code +="""
                settings['value']   = {value}
                """.format(value = self.value_CM)
            code +="""
                {output} = CleanMissingOperation({input},settings,numFrag)
                """.format(output = self.named_outputs['output result'],
                           input  = self.named_inputs['input data'])

            return dedent(code)


class DifferenceOperation(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = (len(named_inputs) == 2) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Difference import DifferenceOperation\n"

    def generate_code(self):
        if self.has_code:

            code = """
            numFrag = 4
            {} = DifferenceOperation({},{},numFrag)
            """.format( self.named_outputs['output data'],
                        self.named_inputs['input data 1'],
                        self.named_inputs['input data 2'])
            return dedent(code)

class DistinctOperation(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """
    ATTRIBUTES_PARAM = 'attributes'

    #se a lista for vazia, entao usar todos

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            self.attributes = []

        if self.has_code:
            self.has_import = "from functions.etl.Distinct import DistinctOperation\n"



    def generate_code(self):
        if self.has_code:
            code = """
            numFrag = 4
            columns = {keys}
            {output} = DistinctOperation({input},columns,numFrag)
            """.format( output=self.named_outputs['output data'],
                        input=self.named_inputs['input data'],
                        keys=self.attributes)
            return dedent(code)


class DropOperation(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Drop import DropOperation\n"

    def generate_code(self):
        if self.has_code:

            code = """
            numFrag = 4
            columns = {columns}
            {output} = DropOperation({input},columns,numFrag)
            """.format(output   = self.named_outputs['output data'],
                       input    = self.named_inputs['input data'],
                       columns  = self.parameters['attributes'])
            return dedent(code)


class FilterOperation(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.FILTER_PARAM not in parameters:
            raise ValueError("Parameter '{}' must be informed for task {}".format(self.FILTER_PARAM, self.__class__))

        tmp = parameters.get(self.FILTER_PARAM)

        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        self.has_import = "from functions.etl.Filter import FilterOperation\n"
        self.query = ""
        for dict in tmp:
            self.query += "({} {} {}) &".format(dict['attribute'], dict['f'], dict['alias'] )
        self.query = self.query[:-2]

        if self.has_code:
            self.generate_code()

    def generate_code(self):
        output = self.named_outputs['output data'] if len(self.named_outputs) else '{}_tmp'.format(self.named_inputs['input data'])

        code = """
        numFrag = 4
        settings = dict()
        settings['query'] = "{query}"
        {out} = FilterOperation({input},settings,numFrag)
        """.format( out   = output,
                    input = self.named_inputs['input data'],
                    query = self.query)


        return dedent(code)


class Intersection(Operation):
    """
    Returns a new DataFrame containing rows only in both this frame and another frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.parameters = parameters
        self.has_code = (len(named_inputs) == 2) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Intersect import IntersectionOperation\n"

    def generate_code(self):

        code = "{} = IntersectionOperation({},{})".format(self.named_outputs['output data'],
                                              self.named_inputs['input data 1'],
                                              self.named_inputs['input data 2'])
        return dedent(code)


class JoinOperation(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM   = 'keep_right_keys'
    MATCH_CASE_PARAM        = 'match_case'
    JOIN_TYPE_PARAM         = 'join_type'
    LEFT_ATTRIBUTES_PARAM   = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM  = 'right_attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.keep_right_keys = parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')
        if self.has_code:
            self.has_import = "from functions.etl.Join import JoinOperation\n"

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.LEFT_ATTRIBUTES_PARAM, self.RIGHT_ATTRIBUTES_PARAM,
                    self.__class__))
        else:
            self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
            self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

    def generate_code(self):
        output = self.named_outputs.get('output data', 'intersected_data_{}'.format(self.order))


        code = """
            numFrag = 4
            params = dict()
            params['option'] = '{type}'
            params['key1']   = {id1}
            params['key2']   = {id2}
            params['case']   = {case}
            params['keep_keys'] = {keep}
            {out} = JoinOperation({in1},{in2}, params, numFrag)
            """.format( out  = output,
                        type = self.join_type,
                        in1  = self.named_inputs['input data 1'],
                        in2  = self.named_inputs['input data 2'],
                        id1  = self.parameters['left_attributes'],
                        id2  = self.parameters['right_attributes'],
                        case = self.match_case,
                        keep = self.keep_right_keys)

        return dedent(code)


class ReplaceValuesOperation(Operation): # ok
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)
        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.ReplaceValues import ReplaceValuesOperation\n"
        self.mode = parameters.get('mode', 'value')
        self.input_regex = False if self.mode == 'value' else True
        self.input_replaces = {}
        i = 0

        if not self.input_regex:
            self.has_code = (len(self.parameters['old_value']) > 0) and (len(self.parameters['new_value']) > 0)

            for att in parameters['attributes']:
                if att not in self.input_replaces:
                    self.input_replaces[att] = [[],[]]
                self.input_replaces[att][0].append(self.parameters['old_value'][i])
                self.input_replaces[att][1].append(self.parameters['new_value'][i])
                i+=1
        else:
            for att in parameters['attributes']:
                if att not in self.input_replaces:
                    self.input_replaces[att] = [[],[]]
                self.input_replaces[att][0].append(self.parameters['regex'][i])
                self.input_replaces[att][1].append(self.parameters['new_value'][i])
                i+=1

    def generate_code(self):

        code = """
            numFrag = 4
            settings = dict()
            settings['replaces'] = {replaces}
            settings['regex'] = {regex}
            {output} = ReplaceValuesOperation({input},settings,numFrag)
            """.format(output = self.named_outputs['output data'],
                       input  = self.named_inputs['input data'],
                       replaces = self.input_replaces,
                       regex = self.input_regex)
        return dedent(code)


class SampleOrPartition(Operation):


    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = (len(self.named_inputs) == 1) and (len(self.named_outputs) > 0)
        if self.has_code:
            self.has_import = "from functions.etl.Sample import SampleOperation\n"

        self.type   = self.parameters.get('type', 'percent')
        self.value  = self.parameters.get('value', None)
        if (self.value == None) or (self.value < 0):
            self.type = 'percent'

        self.seed   = self.parameters.get('seed', None)


    def generate_code(self):
        code = """
        numFrag  = 4
        settings = dict()
        settings['type']  = '{type}'
        settings['value'] = {value}
        settings['seed']  = {seed}
        {output} = SampleOperation({input},settings,numFrag)
        """.format(output= self.named_outputs['sampled data'],
                   input = self.named_inputs['input data'],
                   type  = self.type,
                   seed  = self.seed,
                   value = self.value)
        return dedent(code)




class SelectOperation(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Select import SelectOperation\n"

    def generate_code(self):

        code = """
        numFrag = 4
        columns = [{column}]
        {output} = SelectOperation({input},columns,numFrag)
        """.format( output = self.named_outputs['output projected data'],
                    input = self.named_inputs['input data'],
                    column = ', '.join(['"{}"'.format(x) for x in self.attributes]))
        return dedent(code)


class SortOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        if self.has_code:
            self.has_import = "from functions.etl.Sort import SortOperation\n"

        self.input_columns = [ dict['attribute'] for dict in parameters['attributes'] if dict['attribute'] != None]
        tmp =  [ dict['f'] for dict in parameters['attributes'] if dict['f'] != None]
        if len(self.input_columns) == len(tmp):
            self.order = []
            for v in tmp:
                if v == "asc":
                    self.order.append(True)
                else:
                    self.order.append(False)
        else:
            self.order = True if tmp[0] == 'asc' else False

        self.has_code = ((len(named_inputs) == 1) and len(self.input_columns)>0 and len(self.order)>0)


    def generate_code(self):

        self.algo = "bitonic"  #"odd-even" "bitonic"  => avaliar como receber o numFrag no Juicer

        code = """
            numFrag = 4
            settings = dict()
            settings['columns'] = {columns}
            settings['ascending'] = {asc}
            settings['algorithm'] = '{algo}'
            {output} = SortOperation({input},settings,numFrag)
            """.format(output = self.named_outputs['output data'],
                       input  = self.named_inputs['input data'],
                       columns= self.input_columns,
                       algo   = self.algo,
                       asc    = self.order)
        return dedent(code)


class SplitOperation(Operation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation ('0' means no seed).
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Split import SplitOperation\n"

        self.out1 = self.named_outputs.get('splitted data 1', '{}_1_tmp'.format(self.output))
        self.out2 = self.named_outputs.get('splitted data 2', '{}_2_tmp'.format(self.output))

    def get_data_out_names(self, sep=','):
            return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.out1, self.out2])



    def generate_code(self):
        v = self.parameters.get('weights', 0)
        self.percentage = float(v)/100

        code =  """
        numFrag  = 4
        settings = dict()
        settings['percentage'] = {percentage}
        settings['seed']       = {seed}
        {out1},{out2} = SplitOperation({input},settings,numFrag)
                """.format( out1    = self.out1,
                            out2    = self.out2,
                            input   = self.named_inputs['input data'],
                            seed    = self.parameters.get("seed", None),
                            percentage = self.percentage
                            )
        return dedent(code)

class TransformationOperation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing, replace it.
        - Expression: json describing the transformation expression
    """
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if all(['alias' in parameters, 'expression' in parameters]):
            self.alias = parameters['alias']
            import json
            self.json_expression = json.loads(parameters['expression'])['tree']
        # else:
        #     raise ValueError(
        #         "Parameters '{}' and {} must be informed for task {}".format(
        #             self.ALIAS_PARAM, self.EXPRESSION_PARAM, self.__class__))

        self.has_code = (len(named_inputs) == 1) and (len(self.named_outputs)>0)
        if self.has_code:
            self.has_import = "from functions.etl.Transform import TransformOperation\n"



    def generate_code(self):
        output = self.named_outputs.get('output data', 'sampled_data_{}'.format(self.order))
        input_data = self.named_inputs['input data']

        # Builds the expression and identify the target column
        params = {'input': input_data}
        expression = Expression(self.json_expression, params)

        functions = [ self.alias, expression.parsed_expression, expression.imports ]

        code = """
        numFrag = 4
        settins = dict()
        settings['functions']   = [{expr}]
        {out} = TransformOperation({input}, settings, numFrag)
        """.format(out=output, input=input_data, expr=functions)
        return dedent(code)



class UnionOperation(Operation): #ok
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2 and len(self.named_outputs) > 0
        if self.has_code:
            self.has_import = "from functions.etl.Union import UnionOperation\n"

    def generate_code(self):
        code = """
        numFrag = 4
        {0} = UnionOperation({1},{2}, numFrag)
        """.format( self.named_outputs['output data'],
                    self.named_inputs['input data 1'],
                    self.named_inputs['input data 2'])
        return dedent(code)










