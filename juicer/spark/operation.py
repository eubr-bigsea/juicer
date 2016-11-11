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
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    INFER_SCHEMA_PARAM = 'infer_schema'

    def __init__(self, parameters, inputs, outputs):
        self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
        self.header = parameters.get(self.HEADER_PARAM, True)
        self.sep = parameters.get(self.SEPARATOR_PARAM, ',')
        self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM, True)

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
        self.columns = ast.literal_eval(parameters['columns'])
        self.ascending = map(lambda x: int(x), ast.literal_eval(parameters['ascending']))
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{0} = {1}.orderBy({2}, ascending={3})".format(self.outputs[0],
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
    '''
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        #self.expression = parameters['expression']
        self.column = ast.literal_eval(parameters['column'])
    def generate_code(self):
        code = '{} = {}.join({}, {})'.format(self.outputs[0],
            self.inputs[0],self.inputs[1],self.column)
        return dedent(code)



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
            self.outputs[0], self.outputs[1], self.column)
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
        print self.columns
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
        self.column = parameters['column']
        self.function = parameters['function']
    def generate_code(self):
        info = {self.column: self.function}
        code = '''{} = {}.groupBy('{}').agg({})'''.format(
            self.outputs[0],self.inputs[0], self.column, json.dumps(info))
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




