

class DataReader():
    '''
    Reads a database.
    Parameters:
    - File format (for now, just csv is suported).
    - Boolean value indicating if the file has a header.
    - Delimiter char for the csv file.
    '''
    def __init__(self, parameters, input_df, output_df):
        self.infile = parameters['infile']
        self.header = parameters['has_header']
        self.delimiter = parameters['sep']
        self.input_df = input_df
        self.output_df = output_df
    def generate_code(self):
        spark_code = (self.output_df[0] + " = spark.read.csv('" + self.infile + \
                      "', header=" + self.header + ", sep='" + self.delimiter + "')")
        return spark_code



class RandomSplit():
    '''
    Randomly splits the Data Frame into two data frames.
    Parameters:
    - List with two weights for thw two new data frames.
    - Optional seed in case of deterministic random operation ('0' means no seed).
    '''
    def __init__(self, parameters, input_df, output_df):
        self.weight_1 = parameters['weights'][0]
        self.weight_2 = parameters['weights'][1]
        self.seed = parameters['seed']
        self.input_df = input_df
        self.output_df = output_df
    def generate_code(self):
        spark_code = (self.output_df[0] + " = " + self.input_df[0] + ".randomSplit([" + self.weight_1 + \
                     ", " + self.weight_2 + "], " + self.seed + ")")
        spark_code += "\n" + self.output_df[1] + " = " + self.output_df[0] + "[1]"
        spark_code += "\n" + self.output_df[0] + " = " + self.output_df[0] + "[0]"
        return spark_code



class Union():
    '''
    Return a new DataFrame containing union of rows in this frame and another frame.
    Parameter: boolean distinct indicating if duplicates should be removed.
    '''
    def __init__(self, parameters, input_df, output_df):
        self.distinct = parameters['distinct']
        self.input_df = input_df
        self.output_df = output_df
    def generate_code(self):
        spark_code = (self.output_df[0] + " = " + self.input_df[0] + ".unionAll(" + self.input_df[1] + ")")
        if (self.distinct == "True"):
           spark_code += ".distinct()"
        return spark_code




class Sort():
    ''' 
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of boolean to indicating if it is ascending sorting.
    '''
    def __init__(self, parameters, input_df, output_df):
        self.columns = parameters['distinct']
        self.ascending = parameters['ascending']
        self.input_df = input_df
        self.output_df = output_df
    def generate_code(self):
        spark_code = (self.output_df[0] + " = " + self.input_df[0] + ".orderBy(" + self.columns + \
                     ", ascending = " + self.ascending + ")")
        return spark_code

