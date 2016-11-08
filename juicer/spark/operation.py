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
        #self.header = parameters['header']
        #self.sep = parameters['sep']
        #self.infer_schema = parameters['infer_schema']
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
                "True", ",", "True")

        elif self.metadata['format'] == 'PARQUET_FILE':
            pass
        elif self.metadata['format'] == 'JSON_FILE':
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
    - withReplacement -> can elements be sampled multiple times (replaced when sampled out)
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

    def generate_code(self):
        code = """{0} = spark.read.csv('{1}',
            header=True, sep=',' ,inferSchema=True)""".format(
            self.outputs[0], self.url)
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
        - New columns: boolean indicating if a new columns should be created
        - Alias: new name of the column. If it is empty, keep the same name
        - Expression: json describing the transformation expression
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.new_column = parameters['new_column']
        if parameters['alias'] == "":
            self.alias = None
        else:
            self.alias = parameters['alias']
        self.json_expression = parameters['expression']
        self.built_expression = ''
        self.target_column = ''

        print "\n\n", json.dumps(self.json_expression, indent=3), "\n\n"


    def generate_code(self):

        # First, builds the expression and identify the target column
        expression = Expression(self.json_expression)
        self.built_expression = expression.parsed_expression
        self.target_column = expression.target

        # For testing without using expression parser:
        #self.built_expression = self.json_expression
        #self.target_column = "sex"

        # Transform and replace the existing column with the same name
        if self.new_column == 0 and self.alias == None:
            code = '''{} = {}.withColumn("{}", {})'''.format(self.outputs[0],
                self.inputs[0], self.target_column,self.built_expression)

        # Transform the existing column and rename it
        elif self.new_column == 0 and self.alias != None:
            code = '''{} = {}.withColumn("{}", {}).withColumnRenamed("{}", "{}")
                '''.format(self.outputs[0],self.inputs[0], self.target_column,
                    self.built_expression, self.target_column, self.alias)

        # Create a new column and set a default name for it
        elif self.new_column == 1 and self.alias == None:
            code = '''
                new_name = "column" + str(len({}.columns))
                {} = {}.withColumn(new_name, {})
            '''.format(self.inputs[0], self.outputs[0],self.inputs[0],
                       self.built_expression)

        # Create a new column and set the new name stored in 'alias'
        else:
            code = '''{} = {}.withColumn("{}", {})'''.format(self.outputs[0],
                self.inputs[0], self.alias,self.built_expression)


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

    def generate_code(self):

        if (self.format == "CSV"):
            code_save = """{}.write.csv('{}', header=True)""".format(
                self.inputs[0], self.url)
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




