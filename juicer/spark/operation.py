# coding=utf-8
import ast
import json
#from expression import Expression
from textwrap import dedent
from metadata import MetadataGet
from expression import Expression


class operation():
    def set_io(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class DataReader(operation):
    '''
    Reads a database.
    Parameters:
    - Limonero database ID
    '''
    def __init__(self, parameters, inputs, outputs):
        self.database_id = parameters['database_id']
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = True
        try:
            self.sep = parameters['sep']
        except KeyError:
            self.sep = ","
        try:
            self.infer_schema = parameters['infer_schema']
        except KeyError:
            self.infer_schema = True
        self.set_io(inputs, outputs)
        metadata_obj = MetadataGet('123456')
        self.metadata = metadata_obj.get_metadata(self.database_id)

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero? such as header and sep.
        if self.metadata['format'] == 'CSV':
            code = """{} = spark.read.csv('{}',
            header={}, sep='{}' ,inferSchema={})""".format(
                self.outputs[0], self.metadata['url'],
                self.header, self.sep, self.infer_schema)

        elif self.metadata['format'] == 'PARQUET_FILE':
            # TO DO
            pass
        elif self.metadata['format'] == 'JSON_FILE':
            # TO DO
            pass

        return dedent(code)



class RandomSplit(operation):
    '''
    Randomly splits the Data Frame into two data frames.
    Parameters:
    - List with two weights for thw two new data frames.
    - Optional seed in case of deterministic random operation ('0' means no seed).
    '''
    def __init__(self, parameters, inputs, outputs):
        self.weights = map(lambda x: float(x), ast.literal_eval(parameters['weights']))
        self.seed = parameters['seed']
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = """{0}, {1} = {2}.randomSplit({3}, {4})""".format(
            self.outputs[0], self.outputs[1], self.inputs[0],
            json.dumps(self.weights), self.seed)
        return dedent(code)



class AddLines(operation):
    '''
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters. 
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{0} = {1}.unionAll({2})".format(self.outputs[0], 
            self.inputs[0], self.inputs[1])
        return dedent(code)




class Sort(operation):
    ''' 
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of 
               boolean to indicating if it is ascending sorting.
    '''
    def __init__(self, parameters, inputs, outputs):
        #self.columns = ast.literal_eval(parameters['columns'])
        #self.ascending = map(lambda x: int(x), ast.literal_eval(parameters['ascending']))
        self.columns = map(lambda x: str(x), parameters['columns'])
        self.ascending = map(lambda x: str(x), parameters['ascending'])
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{} = {}.orderBy({}, ascending={})".format(self.outputs[0], 
        self.inputs[0], str(json.dumps(self.columns)), str(json.dumps(self.ascending)))
        return dedent(code)




class Distinct(operation):
    '''
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    No parameters required.
    '''
    def __init__(self, parameters, inputs, outputs):
       self.set_io(inputs, outputs) 
    def generate_code(self):
        code = "{} = {}.distinct()".format(
            self.outputs[0], self.inputs[0])
        return dedent(code)



class Sample(operation):
    '''
    Returns a sampled subset of this DataFrame.
    Parameters:
    - withReplacement -> can elements be sampled multiple times
                        (replaced when sampled out)
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen; 
            fraction must be [0, 1]
        with replacement: expected number of times each element is chosen; 
            fraction must be >= 0
    - seed -> seed for random operation.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.withReplacement = parameters['withReplacement']
        self.fraction = parameters['fraction']
        self.seed = parameters['seed']
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{} = {}.sample(withReplacement={}, fraction={}, seed={})".format(
            self.outputs[0], self.inputs[0], self.withReplacement, 
            self.fraction, self.seed)
        return dedent(code)



class Intersection(operation):
    '''
    Returns a new DataFrame containing rows only in both this frame 
    and another frame.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{} = {}.intersect({})".format(
            self.outputs[0], self.inputs[0], self.inputs[1])
        return dedent(code)



class Difference(operation):
    '''
    Returns a new DataFrame containing rows in this frame but not in another frame.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{} = {}.subtract({})".format(
            self.outputs[0], self.inputs[0], self.inputs[1])
        return dedent(code)



class Join(operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type_param'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.keep_right_keys = parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

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
        on_clause = zip(self.left_attributes, self.right_attributes)
        join_condition = ', '.join([
                                       '{}.{} == {}.{}'.format(self.inputs[0],
                                                               pair[0],
                                                               self.inputs[1],
                                                               pair[1]) for pair
                                       in on_clause])

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[1])

        code = """
            cond_{0} = [{1}]
            {0} = {2}.join({3}, on=cond_{0}, how='{4}')""".format(
            output, join_condition, self.inputs[0], self.inputs[1],
            self.join_type)


        # TO-DO: Convert str False to boolean for evaluation
        if self.keep_right_keys == "False":
            for column in self.right_attributes:
                code += """.drop({}.{})""".format(self.inputs[1], column)

        return dedent(code)


# class Join(operation):
#     '''
#     Joins with another DataFrame, using the given join expression.
#     Parameters:
#         - Columns left: name of the columns from the first dataframe
#         - Columns right: name of the columns from the second dataframe
#         - Join type: inner, outer, left_outer, right_outer or leftsemi (Default: inner)
#     '''
#     def __init__(self, parameters, inputs, outputs):
#         self.set_io(inputs, outputs)
#         self.columns_left = map(lambda x: str(x), parameters['columns_left'])
#         self.columns_right = map(lambda x: str(x), parameters['columns_right'])
#         self.type = parameters.get('type','inner')
#     def generate_code(self):
#         expressions = []
#         for i in range(0,len(self.columns_left)):
#             expressions.append("""({}.{} == {}.{})""".format(self.inputs[0],
#                     self.columns_left[i], self.inputs[1], self.columns_right[i]))
#         code = """{} = {}.join({}, {}, '{}')""".format(self.outputs[0],
#             self.inputs[0],self.inputs[1],' & '.join(expressions), self.type)
#
#         return dedent(code)



class ReadCSV(operation):
    '''
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
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
        code = """{} = spark.read.csv('{}',
            header={}, sep='{}' ,inferSchema=True)""".format(
            self.outputs[0], self.url, self.header, self.separator)
        return dedent(code)



class Drop(operation):
    '''
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.column = parameters['column']
    def generate_code(self):
        code = """{} = {}.drop('{}')""".format(
            self.outputs[0], self.inputs[0], self.column)
        return dedent(code)



class Transformation(operation):
    '''
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing, replace it.
        - Expression: json describing the transformation expression
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.alias = parameters['alias']
        self.json_expression = parameters['expression']
    def generate_code(self):
        # Builds the expression and identify the target column
        expression = Expression(self.json_expression)
        built_expression = expression.parsed_expression
        # Builds the code
        code = '''{} = {}.withColumn('{}', {})'''.format(self.outputs[0],
            self.inputs[0], self.alias,built_expression)
        return dedent(code)



class Select(operation):
    '''
    Projects a set of expressions and returns a new DataFrame.
    Paramaters:
    - The list of columns selected.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.columns = map(lambda x: str(x), parameters['columns'])
    def generate_code(self):
        code = '''{} = {}.select({})'''.format(
            self.outputs[0],self.inputs[0], self.columns)
        return dedent(code)



class Aggregation(operation):
    '''
    Compute aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum, count.
    '''

    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.group_by = map(lambda x: str(x), parameters['group_by'])
        self.columns = map(lambda x: str(x), parameters['columns'])
        self.function = map(lambda x: str(x), parameters['functions'])
        self.names = map(lambda x: str(x), parameters['new_names'])
    def generate_code(self):
        elements = []
        for i in range(0, len(self.columns)):
            content = '''{}('{}').alias('{}')'''.format(self.function[i], self.columns[i],
                           self.names[i])
            elements.append(content)
        code = '''{} = {}.groupBy({}).agg({})'''.format(
            self.outputs[0],self.inputs[0], self.group_by, ', '.join(elements))

        #info = {self.column: self.function}
        #code = '''{} = {}.groupBy('{}').agg({})'''.format(
        #    self.outputs[0],self.inputs[0], self.column, json.dumps(info))
        return dedent(code)



class Filter(operation):
    '''
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.expression = parameters['expression']
    def generate_code(self):
        code = '''{} = {}.filter('{}')'''.format(
            self.outputs[0], self.inputs[0], self.expression)
        return dedent(code)


class DatetimeToBins(operation):
    '''
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.target_column = parameters['target_column']
        self.new_column = parameters['new_column']
        self.group_size = parameters['group_size']
    def generate_code(self):
        code = '''
            from bins import *
            {} = datetime_to_bins({}, {}, '{}', '{}')
            '''.format(self.outputs[0], self.inputs[0], self.group_size, 
		self.target_column, self.new_column)
        return dedent(code)




class Save(operation):
    '''
    Saves the content of the DataFrame at the specified path
    and generate the code to call the Limonero API.
    Parameters:
        - Database name
        - URL for storage
        - Storage ID
        - Database tags
        - Workflow that generated the database
    '''
    def __init__(self, parameters, inputs, outputs):
        self.name = parameters['name']
        self.format = parameters['format']
        self.url = parameters['url']
        self.storage_id = parameters['storage_id']
        self.tags =  ast.literal_eval(parameters['tags'])
        self.set_io(inputs, outputs)
        self.workflow = parameters['workflow']
        try:
	    self.mode = parameters['mode']
        except KeyError:
	    self.mode = "error"
        try:
	    self.header = parameters['header']
        except KeyError:
	    self.header = "True"



    def generate_code(self):

        if (self.format == "CSV"):
            code_save = """{}.write.csv('{}', header={}, mode='{}')""".format(
                self.inputs[0], self.url, self.header, self.mode)
        elif (self.format == "PARQUET"):
            pass
        elif (self.format == "JSON"):
            pass

        code_api = """
            from metadata import MetadataPost

            types_names = dict()
            types_names['IntegerType'] = "INTEGER"
            types_names['StringType'] = "TEXT"
            types_names['LongType'] = "LONG"
            types_names['DoubleType'] = "DOUBLE"
            types_names['TimestampType'] = "DATETIME"


            schema = []
            for att in {0}.schema:
                data = dict()
                data['name'] = att.name
                data['dataType'] = types_names[str(att.dataType)]
                data['nullable'] = att.nullable
                data['metadata'] = att.metadata
                schema.append(data)

            parameters = dict()
            parameters['name'] = "{1}"
            parameters['format'] = "{2}"
            parameters['storage_id'] = {3}
            parameters['provenience'] = str("{4}")
            parameters['description'] = "{5}"
            parameters['user_id'] = "{6}"
            parameters['user_login'] = "{7}"
            parameters['user_name'] = "{8}"
            parameters['workflow_id'] = "{9}"
            parameters['url'] = "{10}"

            instance = MetadataPost('{11}', schema, parameters)
            """.format(self.inputs[0], self.name, self.format, self.storage_id,
                str(json.dumps(self.workflow)).replace("\"","'"),
                self.workflow['workflow']['name'], self.workflow['user']['id'],
                self.workflow['user']['login'],self.workflow['user']['name'],
                self.workflow['workflow']['id'], self.url, "123456"
            )

        code = dedent(code_save) + dedent(code_api)

        return code




