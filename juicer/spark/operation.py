from textwrap import dedent
import json



class operation():
    def set_io(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs



class DataReader(operation):
    '''
    Reads a database.
    Parameters:
    - File format (for now, just csv is suported).
    - Boolean value indicating if the file has a header.
    - Delimiter char for the csv file.
    '''
    def __init__(self, parameters, inputs, outputs):
        self.infile = parameters['infile']
        self.header = parameters['has_header']
        self.delimiter = parameters['sep']
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = """{0} = spark.read.csv('{1}', header={2}, sep='{3}')""".format(
            self.outputs[0], self.infile, self.header, self.delimiter)
        return dedent(code)



class RandomSplit(operation):
    '''
    Randomly splits the Data Frame into two data frames.
    Parameters:
    - List with two weights for thw two new data frames.
    - Optional seed in case of deterministic random operation ('0' means no seed).
    '''
    def __init__(self, parameters, inputs, outputs):
        self.weights = map(lambda x: float(x), parameters['weights'])
        self.seed = parameters['seed']
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = """{0}, {1} = {2}.randomSplit({3}, {4})""".format(
            self.outputs[0], self.outputs[1], self.inputs[0],
            json.dumps(self.weights), self.seed)
        return dedent(code)



class Union(operation):
    '''
    Return a new DataFrame containing union of rows in this frame and another frame.
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
        self.columns = parameters['columns']
        self.ascending = map(lambda x: int(x), parameters['ascending'])
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{0} = {1}.orderBy({2}, ascending={3})".format(self.outputs[0], 
        self.inputs[0], str(json.dumps(self.columns)), str(json.dumps(self.ascending)))
        return dedent(code)



# FALTA TESTAR COM SPARK
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



# FALTA TESTAR COM SPARK
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


# FALTA TESTAR
class Save(operation):
    '''
    Saves the content of the DataFrame at the specified path.
    Parameters:
    - Format -> Supported formats: CSV, Json and Parquet.
      CSV files must have header, comma as separator and quotation marks for strings
    - Path -> the path in any Hadoop supported file system
    - Mode -> specifies the behavior of the save operation when data already exists.
              (append or overwrite or ignore or error)
    - Compression ->  compression codec to use when saving to file.
                      (none, bzip2, gzip, lz4, snappy and deflate)   
    '''
    def __init__(self, parameters, inputs, outputs):
        self.path = parameters['path']
        self.file_format = parameters['format']
        self.mode = parameters['mode']
#        if parameters.has_key['mode']:
#            self.mode = parameters['mode']
#        else:
#            self.mode = 'error'
#        if parameters.has_key('compression'):
#            self.compression = parameters['compression']
#        else:
#            self.compression = None
        self.set_io(inputs, outputs)
    def generate_code(self):
        code = "{}.write.format('{}').mode('{}').save('{}')".format(
            self.inputs[0], self.file_format, self.mode, self.path)
#        code = "{}.write.csv('{}')".format(
#            self.inputs[0], self.path)
        return dedent(code)



