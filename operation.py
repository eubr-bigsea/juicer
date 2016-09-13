class Operation():
    pass

class sparkOperation(Operation):
    pass

class spark_randomSplit(sparkOperation):
    

from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("## Lemonade_workflow_consumer ##")\
    .getOrCreate()

def random_split(task, response):
    
    splitResult = dataFrame.randomSplit([1.0,3.0], seed)
    this_response = {}
    response.append()


