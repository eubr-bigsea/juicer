# -*- coding: utf-8 -*-
import json
import zipfile
from textwrap import dedent

import sys

import jinja2
import juicer.spark.data_operation
import juicer.spark.etl_operation
import juicer.spark.geo_operation
import juicer.spark.ml_operation
import juicer.spark.statistic_operation
import juicer.spark.text_operation
import os
from juicer.jinja2_custom import AutoPep8Extension
from juicer.spark import operation
from juicer.util import sort_topologically


class SparkTranspiler:
    """
    Convert Lemonada workflow representation (JSON) into code to be run in
    Apache Spark.
    """
    DIST_ZIP_FILE = '/tmp/lemonade-lib-python.zip'

    def __init__(self, workflow, out=None):
        self.out = sys.stdout if out is None else out
        self.workflow = workflow

        graph = {}
        all_tasks = {}
        for task in workflow['tasks']:
            graph[task['id']] = []
            all_tasks[task['id']] = task
        for flow in workflow['flows']:
            graph[flow['target_id']].append(flow['source_id'])

        dependency = sort_topologically(graph)
        self.tasks = [all_tasks[item] for sublist in dependency for item in
                      sublist]

        self.operations = {}

        self._assign_operations()
        self.execute_main = False

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

    def transpile(self):
        """ Transpile the tasks from Lemonade's workflow into Spark code """

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
                ports[source_id] = {'outputs': [], 'inputs': [],
                                    'named_inputs': {}, 'named_outputs': {}}
            if target_id not in ports:
                ports[target_id] = {'outputs': [], 'inputs': [],
                                    'named_inputs': {}, 'named_outputs': {}}

            sequence = sequential_ports[flow_id]
            if sequence not in ports[source_id]['outputs']:
                ports[source_id]['named_outputs'][
                    flow['source_port_name']] = sequence
                ports[source_id]['outputs'].append(sequence)
            if sequence not in ports[target_id]['inputs']:
                ports[target_id]['named_inputs'][
                    flow['target_port_name']] = sequence
                ports[target_id]['inputs'].append(sequence)

        env_setup = {'instances': [],
                     'workflow_name': self.workflow.get('name'),
                     }
        workflow_json = json.dumps(self.workflow)
        for i, task in enumerate(self.tasks):
            ##self.output.write("\n# {}\n".format(task['operation']['name']))
            # input_list = []
            # output_list = []
            # self.map_port(task, input_list, output_list)
            class_name = self.operations[task['operation']['slug']]

            parameters = {}
            # print task['forms']
            for parameter, definition in task['forms'].iteritems():
                # @FIXME: Fix wrong name of form category
                # (using name instead of category)
                # print definition.get('category')
                # raw_input()
                cat = definition.get('category',
                                     'execution').lower()  # FIXME!!!
                cat = 'paramgrid' if cat == 'param grid' else cat
                cat = 'logging' if cat == 'execution logging' else cat

                if all([cat in ["execution", 'paramgrid', 'param grid',
                                'execution logging', 'logging'],
                        definition['value'] is not None]):
                    # print '###{} ==== {}'.format(parameter, definition['value'])

                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']

            # Operation SAVE requires the complete workflow
            if task['operation']['name'] == 'SAVE':
                parameters['workflow'] = self.workflow

            # Some temporary variables need to be identified by a sequential
            # number, so it will be stored in this field
            task['order'] = i

            parameters['task'] = task
            parameters['workflow_json'] = workflow_json
            parameters['user'] = self.workflow.get('user', {})
            parameters['workflow_id'] = self.workflow.get('id')
            port = ports.get(task['id'], {})
            instance = class_name(parameters, port.get('inputs', []),
                                  port.get('outputs', []),
                                  port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))
            # if instance.has_code:
            ## self.output.write(instance.generate_code() + "\n")
            env_setup['instances'].append(instance)
            env_setup['execute_main'] = self.execute_main

            # Just for testing. Remove from here.0
            # for out in output_list:
            #    self.output.write(
            #        "print \"" + task['operation']['name'] + "\" \n")
            # self.output.write(out + ".show()\n")
            # Until here.
        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension])
        template = template_env.get_template("operation.tmpl")
        self.out.write(template.render(env_setup))

    def _assign_operations(self):
        self.operations = {
            'add-columns': juicer.spark.etl_operation.AddColumns,
            'add-rows': juicer.spark.etl_operation.AddRows,
            'aggregation': juicer.spark.etl_operation.Aggregation,
            'apply-model': juicer.spark.ml_operation.ApplyModel,
            'change-attribute': juicer.spark.data_operation.ChangeAttribute,
            'clean-missing': juicer.spark.etl_operation.CleanMissing,
            'classification-model':
                juicer.spark.ml_operation.ClassificationModel,
            'classification-report':
                juicer.spark.ml_operation.ClassificationReport,
            'clustering-model':
                juicer.spark.ml_operation.ClusteringModelOperation,
            'comment': operation.NoOp,
            'cross-validation':
                juicer.spark.ml_operation.CrossValidationOperation,
            'data-reader': juicer.spark.data_operation.DataReader,
            'data-writer': juicer.spark.data_operation.Save,
            'decision-tree-classifier':
                juicer.spark.ml_operation.DecisionTreeClassifierOperation,
            'difference': juicer.spark.etl_operation.Difference,
            'distinct': juicer.spark.etl_operation.Distinct,
            'drop': juicer.spark.etl_operation.Drop,
            'evaluate-model': juicer.spark.ml_operation.EvaluateModel,
            'feature-assembler': juicer.spark.ml_operation.FeatureAssembler,
            'feature-indexer': juicer.spark.ml_operation.FeatureIndexer,
            'filter': juicer.spark.etl_operation.Filter,
            'read-shapefile': juicer.spark.geo_operation.ReadShapefile,
            'within': juicer.spark.geo_operation.GeoWithin,
            # Alias for filter
            'filter-selection': juicer.spark.etl_operation.Filter,
            'gaussian-mixture-clustering':
                juicer.spark.ml_operation.GaussianMixtureClusteringOperation,
            'generate-n-grams':
                juicer.spark.text_operation.GenerateNGramsOperation,
            'gbt-classifier': juicer.spark.ml_operation.GBTClassifierOperation,
            'intersection': juicer.spark.etl_operation.Intersection,
            'join': juicer.spark.etl_operation.Join,
            'k-means-clustering':
                juicer.spark.ml_operation.KMeansClusteringOperation,
            'lda-clustering': juicer.spark.ml_operation.LdaClusteringOperation,
            'naive-bayes-classifier':
                juicer.spark.ml_operation.NaiveBayesClassifierOperation,
            'pearson-correlation':
                juicer.spark.statistic_operation.PearsonCorrelation,
            'perceptron-classifier':
                juicer.spark.ml_operation.PerceptronClassifier,
            # synonym for select
            'projection': juicer.spark.etl_operation.Select,
            'random-forest-classifier':
                juicer.spark.ml_operation.RandomForestClassifierOperation,
            'read-csv': juicer.spark.data_operation.ReadCSV,
            'replace': juicer.spark.etl_operation.Replace,
            # synonym for distinct
            'remove-duplicated-rows': juicer.spark.etl_operation.Distinct,
            'remove-stop-words':
                juicer.spark.text_operation.RemoveStopWordsOperation,
            'sample': juicer.spark.etl_operation.SampleOrPartition,
            'save': juicer.spark.data_operation.Save,
            'select': juicer.spark.etl_operation.Select,
            # synonym of intersection'
            'set-intersection': juicer.spark.etl_operation.Intersection,
            'sort': juicer.spark.etl_operation.Sort,
            'split': juicer.spark.etl_operation.RandomSplit,
            'svm-classification':
                juicer.spark.ml_operation.SvmClassifierOperation,
            'tokenizer': juicer.spark.text_operation.TokenizerOperation,
            'topic-report': juicer.spark.ml_operation.TopicReportOperation,
            'transformation': juicer.spark.etl_operation.Transformation,
            'word-to-vector': juicer.spark.text_operation.WordToVectorOperation
        }
