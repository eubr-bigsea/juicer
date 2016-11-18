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
        """ Print the PySpark header and session init  """
        code = """
        from pyspark.sql.functions import *
        from pyspark.sql.window import Window
        from pyspark.sql.types import *
        from pyspark.sql import SparkSession
        spark = SparkSession \\
            .builder \\
            .appName('## {} ##') \\
            .getOrCreate()
        """.format(self.workflow['name'])
        self.output.write(dedent(code))

    def map_port(self, task, input_list, output_list):
        """ Map each port of a task to a dict """
        for port in task['ports']:
            if port['interface'] == "dataframe":
                # If port is out, create a new data frame and increment the counter
                if port['direction'] == 'out':
                    self.dataframes[port['id']] = self.workflow['name'] + \
                                                  '_df_' + str(
                        self.count_dataframes)
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
        """ Executes the tasks in Lemonade's workflow """

        ports = {}
        sequential_ports = {}

        for flow in self.workflow['flows']:
            source_id = flow['source_id']
            target_id = flow['target_id']

            flow_id = '[{}:{}]=>[{}:{}]'.format(source_id,
                                                flow['source_port'],
                                                target_id,
                                                flow['target_port'])

            if flow_id not in sequential_ports:
                sequential_ports[flow_id] = 'df_{}'.format(
                    len(sequential_ports))

            if source_id not in ports:
                ports[source_id] = {'outputs': [], 'inputs': []}
            if target_id not in ports:
                ports[target_id] = {'outputs': [], 'inputs': []}

            ports[source_id]['outputs'].append(sequential_ports[flow_id])
            ports[target_id]['inputs'].append(sequential_ports[flow_id])

        for task in self.tasks:
            self.output.write("\n# {}\n".format(task['operation']['name']))
            # input_list = []
            # output_list = []
            # self.map_port(task, input_list, output_list)
            class_name = self.classes[task['operation']['slug']]

            parameters = {}
            for parameter, definition in task['forms'].iteritems():
                if all([definition.get('category',
                                       'execution').lower() == "execution",
                        definition['value'] is not None]):
                    parameters[parameter] = definition['value']

            # Operation SAVE requires the complete workflow
            if task['operation']['name'] == 'SAVE':
                parameters['workflow'] = self.workflow

            instance = class_name(parameters,
                                  ports.get(task['id'], {}).get('inputs', []),
                                  ports.get(task['id'], {}).get('outputs', []))
            if instance.has_code:
                self.output.write(instance.generate_code() + "\n")

                # Just for testing. Remove from here.
                # for out in output_list:
                #    self.output.write(
                #        "print \"" + task['operation']['name'] + "\" \n")
                # self.output.write(out + ".show()\n")
                # Until here.

    def assign_operations(self):
        self.classes = {
            'add-columns': operation.AddColumns,
            'add-rows': operation.AddRows,
            'aggregation': operation.Aggregation,
            'clean-missing': operation.CleanMissing,
            'comment': operation.NoOp,
            'data-reader': operation.DataReader,
            'difference': operation.Difference,
            'distinct': operation.Distinct,
            'drop': operation.Drop,
            'evaluate-model': operation.EvaluateModel,
            'filter': operation.Filter,
            'intersection': operation.Intersection,
            'join': operation.Join,
            'pearson-correlation': operation.PearsonCorrelation,
            # synonym for select
            'projection': operation.Select,
            'split': operation.RandomSplit,
            'read-csv': operation.ReadCSV,
            # synonym for distinct
            'replace': operation.Replace,
            'remove-duplicated-rows': operation.Distinct,
            'sample': operation.Sample,
            'save': operation.Save,
            'select': operation.Select,
            # synonym of intersection'
            'set-intersection': operation.Intersection,
            'sort': operation.Sort,
            'svm-classification': operation.SvmClassification,
            'transformation': operation.Transformation,

        }
