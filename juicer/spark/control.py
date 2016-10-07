from juicer.spark import operation


class Spark:

    def __init__(self, outfile, workflow_name, tasks):
        self.output = open(outfile, "w")
        self.workflow_name = workflow_name
        self.print_session()
        self.tasks = tasks
        # Store the name of the dataframe in each port
        self.dataframes = {}
        self.count_dataframes = 0

        self.port_models = {}
        self.classes = {}
        self.assign_operations()


    def print_session(self):
        ''' Print the PySpark header and session init  '''
        self.output.write("from pyspark.sql import SparkSession \n\n")
        self.output.write("spark = SparkSession\\\n")
        self.output.write("    .builder\\\n")
        self.output.write("    .appName('## "+self.workflow_name+" ##')\\\n")
        self.output.write("    .getOrCreate()\n\n")


    def map_port(self, task, input_list, output_list):
        ''' Map each port of a task to a dict '''

        for port in task['ports']:
            if port['interface'] == "dataframe":

                # If port is out, create a new data frame and increment the coubter
                if port['type'] == 'out':

                    self.dataframes[port['id']] = self.workflow_name + \
                        '_df_'+str(self.count_dataframes)
                    output_list.append(self.dataframes[port['id']])
                    self.count_dataframes += 1

                # IF PORT IS IN, RETRIEVE THE DATAFRAME NAME
                else:
                    input_list.append(self.dataframes[port['id']])

            elif port['interface'] == "model":
                # Implement!
                pass

            else:
                # Implement!
                pass


    def execution(self):
        ''' Executes the tasks in Lemonade's workflow '''

        for task in self.tasks:
            self.output.write("\n# " + task['operation_name'] + "\n")

            input_list = []
            output_list = []
            self.map_port(task, input_list, output_list)

            print task['operation_name'], input_list, output_list

            class_name = self.classes[task['operation_name']]
            instance = class_name(task['parameters'][0], input_list, output_list)

            self.output.write(instance.generate_code() + "\n")

            for out in output_list:
                self.output.write("print \"" + task['operation_name'] + "\" \n")
                self.output.write(out + ".show()\n")


    def assign_operations(self):
        self.classes['DATA_READER'] = operation.DataReader
        self.classes['RANDOM_SPLIT'] = operation.RandomSplit
        self.classes['UNION'] = operation.Union
        self.classes['SORT'] = operation.Sort
        self.classes['SAVE'] = operation.Save
        self.classes['DISTINCT'] = operation.Distinct
        self.classes['SAMPLE'] = operation.Sample
        self.classes['INTERSECTION'] = operation.Intersection
        self.classes['DIFFERENCE'] = operation.Difference
