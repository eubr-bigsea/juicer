import zipfile

import os

import jinja2
import juicer.spark.ml_operation
from juicer.spark import operation
from textwrap import dedent


class Spark:
    DIST_ZIP_FILE = '/tmp/lemonade-lib-python.zip'

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
        """.format(self.workflow['name'].encode('utf8'))
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

    def build_dist_file(self):
        """
        Build a Zip file containing files in dist packages. Such packages
        contain code to be executed in the Spark cluster and should be
        distributed among all nodes.
        """
        project_base = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..')

        lib_paths = [
            os.path.join(project_base, 'spark/dist'),
            os.path.join(project_base, 'dist')
        ]
        build = os.path.exists(self.DIST_ZIP_FILE)
        while not build:
            for lib_path in lib_paths:
                dist_files = os.listdir(lib_path)
                zip_mtime = os.path.getmtime(self.DIST_ZIP_FILE)
                for f in dist_files:
                    if zip_mtime < os.path.getmtime(os.path.join(lib_path, f)):
                        build = True
                        break
                if build:
                    break
            build = build or False

        if build:
            zf = zipfile.PyZipFile(self.DIST_ZIP_FILE, mode='w')
            for lib_path in lib_paths:
                zf.writepy(lib_path)
            zf.close()

    def execution(self):
        """ Executes the tasks in Lemonade's workflow """

        ports = {}
        sequential_ports = {}

        for flow in self.workflow['flows']:
            source_id = flow['source_id']
            target_id = flow['target_id']

            flow_id = '[{}:{}]'.format(source_id, flow['source_port'], )

            if flow_id not in sequential_ports:
                sequential_ports[flow_id] = 'df{}'.format(
                    len(sequential_ports))

            if source_id not in ports:
                ports[source_id] = {'outputs': [], 'inputs': []}
            if target_id not in ports:
                ports[target_id] = {'outputs': [], 'inputs': []}

            sequence = sequential_ports[flow_id]
            if sequence not in ports[source_id]['outputs']:
                ports[source_id]['outputs'].append(sequence)
            if sequence not in ports[target_id]['inputs']:
                ports[target_id]['inputs'].append(sequence)

        env_setup = {'instances': [],
                     'workflow_name': self.workflow.get('name')}
        for task in self.tasks:
            ##self.output.write("\n# {}\n".format(task['operation']['name']))
            # input_list = []
            # output_list = []
            # self.map_port(task, input_list, output_list)
            class_name = self.classes[task['operation']['slug']]

            parameters = {}
            for parameter, definition in task['forms'].iteritems():
                if all([definition.get('category',
                                       'execution').lower() == "execution",
                        definition['value'] is not None]):
                    #print '###{} ==== {}'.format(parameter, definition['value'])
                    parameters[parameter] = definition['value']

            # Operation SAVE requires the complete workflow
            if task['operation']['name'] == 'SAVE':
                parameters['workflow'] = self.workflow

            parameters['task'] = task
            instance = class_name(parameters,
                                  ports.get(task['id'], {}).get('inputs', []),
                                  ports.get(task['id'], {}).get('outputs', []))
            if instance.has_code:
                ## self.output.write(instance.generate_code() + "\n")
                env_setup['instances'].append(instance)

                # Just for testing. Remove from here.
                # for out in output_list:
                #    self.output.write(
                #        "print \"" + task['operation']['name'] + "\" \n")
                # self.output.write(out + ".show()\n")
                # Until here.
        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("operation.tmpl")
        print template.render(env_setup).encode('utf8')

    def assign_operations(self):
        self.classes = {
            'add-columns': operation.AddColumns,
            'add-rows': operation.AddRows,
            'aggregation': operation.Aggregation,
            'clean-missing': operation.CleanMissing,
            'comment': operation.NoOp,
            'data-reader': operation.DataReader,
            'data-writer': operation.Save,
            'save': operation.Save,
            'difference': operation.Difference,
            'distinct': operation.Distinct,
            'drop': operation.Drop,
            'evaluate-model': juicer.spark.ml_operation.EvaluateModel,
            'filter': operation.Filter,
            'intersection': operation.Intersection,
            'join': operation.Join,
            'pearson-correlation': operation.PearsonCorrelation,
            # synonym for select
            'projection': operation.Select,
            'split': operation.RandomSplit,
            'read-csv': operation.ReadCSV,
            'replace': operation.Replace,
            # synonym for distinct
            'remove-duplicated-rows': operation.Distinct,
            'sample': operation.Sample,
            'save': operation.Save,
            'select': operation.Select,
            # synonym of intersection'
            'set-intersection': operation.Intersection,
            'sort': operation.Sort,
            'svm-classification': juicer.spark.ml_operation.SvmClassification,
            'transformation': operation.Transformation,

        }
