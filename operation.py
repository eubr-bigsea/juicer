
class DataReader():
    def __init__(self, parameters, input_df, output_df):
        self.infile = parameters['infile']
        self.header = parameters['has_header']
        self.delimiter = parameters['sep']
        self.input_df = input_df
        self.output_df = output_df
    def generate_code(self):
        spark_code = (self.output_df[0] + " = spark.read.csv('" + self.infile + "', header=" + self.header + ", sep='" + self.delimiter + "')")
        return spark_code


class RandomSplit():
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
    def __init__(self, parameters, input_df, output_df):
        self.distinct = parameters['distinct']
        self.input_df = input_df
        self.output_df = output_df
    def generate_code(self):
        spark_code = (self.output_df[0] + " = " + self.input_df[0] + ".unionAll(" + self.input_df[1] + ")")
        if (self.distinct == "True"):
           spark_code += ".distinct()"
        return spark_code
