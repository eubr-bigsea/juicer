import ast
import json
from textwrap import dedent
from metadata import MetadataGet


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
        self.set_io(inputs, outputs)
        metadata_obj = MetadataGet('123456')
        self.metadata = metadata_obj.get_metadata(self.database_id)
        # Remove the next line when port 9000 is added to limonero instance
        self.metadata['url'] = 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/samples/titanic2.csv'

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero? such as header and sep.
        if self.metadata['format'] == 'CSV_FILE':
            code = """{0} = spark.read.csv('{1}',
            header=True, sep=',' ,inferSchema=True)""".format(
                self.outputs[0], self.metadata['url'])

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



class Save(operation):
    '''
    Saves the content of the DataFrame at the specified path.
    Parameters:
    '''
    def __init__(self, parameters, inputs, outputs):
        self.name = parameters['name']
        self.format = parameters['format']
        self.id = parameters['id']
        self.tags =  ast.literal_eval(parameters['tags'])
        self.set_io(inputs, outputs)

    def generate_code(self):

        code = """
            from metadata import MetadataPost
            schema = []
            for att in {0}.schema:
                data = dict()
                data['name'] = att.name
                data['dataType'] = att.dataType
                data['nullable'] = att.nullable
                data['metadata'] = att.metadata
                schema.append(data)
            parameters = dict()
            parameters['name'] = ''
            parameters['format'] = ''
            parameters['provenience'] = ''
            parameters['storage_id'] = ''
            parameters['description'] = ''
            parameters['user_id'] = ''
            parameters['user_login'] = ''
            parameters['user_name'] = ''
            parameters['workflow_id'] = ''
            instance = MetadataPost('123456', schema, parameters)""".format(self.inputs[0])


        # O codigo spark vai usar a classe metadata para fazer o request
        return dedent(code)




