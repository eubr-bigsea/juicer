
class DataReader():
    def __init__(self,infile):
        self.infile = infile
    def read_csv(self, has_header, sep_char, spark):
        return spark.read.csv(self.infile, header=has_header, sep=sep_char)


class RandomSplit():

    def __init__(self, weights, seed):
        self.weight_1 = weights[0]
        self.weight_2 = weights[1]
        self.seed = seed
 
    def split(self, dataframe):
        return dataframe.randomSplit([self.weight_1,self.weight_2], self.seed)



