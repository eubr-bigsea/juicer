from juicer.spark import operation
from textwrap import dedent

class Spark:

    def __init__(self, outfile, workflow, tasks):
        self.output = open(outfile, "w")
        self.workflow = workflow
        self.print_session()
        # Sorted tasks! Do not use the workflow tasks
        self.tasks = tasks
        # Store the name of the dataframe in each port
        self.dataframes = {}
        self.count_dataframes = 0
        self.classes = {}
        self.assign_operations()


    def print_session(self):
        ''' Print the PySpark header and session init  '''
        code = """
        from pyspark.sql.functions import *
        from pyspark.sql.types import *
        from pyspark.sql import SparkSession
        spark = SparkSession \\
            .builder \\
            .appName('## {} ##') \\
            .getOrCreate()
        """.format(self.workflow['workflow']['name'])
        self.output.write(dedent(code))


    def map_port(self, task, input_list, output_list):
        ''' Map each port of a task to a dict '''
        for port in task['ports']:
            if port['interface'] == "dataframe":
                # If port is out, create a new data frame and increment the counter
                if port['direction'] == 'out':
                    self.dataframes[port['id']] = self.workflow['workflow']['name'] + \
                        '_df_'+str(self.count_dataframes)
                    output_list.append(self.dataframes[port['id']])
                    self.count_dataframes += 1
                # If port is in, just retrieve the name of the existing dataframe
                else:
                    input_list.append(self.dataframes[port['id']])

            # For now, the only interface is dataframe. In the future,
            # others, such as models, should be implemented
            elif port['interface'] == "model":
                # Implement!
                pass
            else:
                # Implement!
                pass


    def execution(self):
        ''' Executes the tasks in Lemonade's workflow '''
        for task in self.tasks:
            self.output.write("\n# " + task['operation']['name'] + "\n")
            input_list = []
            output_list = []
            self.map_port(task, input_list, output_list)
            class_name = self.classes[task['operation']['name']]

            parameters = {}
            for parameter in task['parameters']:
                if parameter['category'] == "EXECUTION":
                    parameters[parameter['name']] = parameter['value']

            # Operation SAVE requires the complete workflow
            if task['operation']['name'] == 'SAVE':
                parameters['workflow'] = self.workflow

            instance = class_name(parameters, input_list, output_list)
            self.output.write(instance.generate_code() + "\n")

            # Just for testing. Remove from here.
            for out in output_list:
                self.output.write("print \"" + task['operation']['name'] + " " + task['id'] + "\" \n")
                self.output.write(out + ".show()\n")
            # Until here.


    def assign_operations(self):
        self.classes['DATA_READER'] = operation.DataReader
        self.classes['RANDOM_SPLIT'] = operation.RandomSplit
        self.classes['ADD_LINES'] = operation.AddLines
        self.classes['SORT'] = operation.Sort
        self.classes['SAVE'] = operation.Save
        self.classes['DISTINCT'] = operation.Distinct
        self.classes['SAMPLE'] = operation.Sample
        self.classes['INTERSECTION'] = operation.Intersection
        self.classes['DIFFERENCE'] = operation.Difference
        self.classes['JOIN'] = operation.Join
        self.classes['READ_CSV'] = operation.ReadCSV
        self.classes['DROP'] = operation.Drop
        self.classes['TRANSFORMATION'] = operation.Transformation
        self.classes['SELECT'] = operation.Select
        self.classes['AGGREGATION'] = operation.Aggregation
        self.classes['FILTER'] = operation.Filter
        self.classes['DATETIME_TO_BINS'] = operation.DatetimeToBins



