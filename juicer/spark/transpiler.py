# -*- coding: utf-8 -*-
import json
import pdb
import sys
import zipfile

import jinja2
import juicer.spark.data_operation
import juicer.spark.etl_operation
import juicer.spark.geo_operation
import juicer.spark.ml_operation
import juicer.spark.statistic_operation
import juicer.spark.text_operation
import juicer.spark.ws_operation
import juicer.spark.vis_operation
import networkx as nx
import os
from juicer import operation
from juicer.util.jinja2_custom import AutoPep8Extension


class DependencyController:
    """ Evaluates if a dependency is met when generating code. """

    def __init__(self, requires):
        self._satisfied = set()
        self.requires = requires

    def satisfied(self, _id):
        self._satisfied.add(_id)

    def is_satisfied(self, _id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


class SparkTranspilerVisitor:
    def __init__(self):
        pass

    def visit(self, workflow, operations, params):
        raise NotImplementedError()


class RemoveTasksWhenMultiplexingVisitor(SparkTranspilerVisitor):
    def visit(self, graph, operations, params):

        return graph
        external_input_op = juicer.spark.ws_operation.MultiplexerOperation
        for task_id in graph.node:
            task = graph.node[task_id]
            op = operations.get(task.get('operation').get('slug'))
            if op == external_input_op:
                # Found root
                if params.get('service'):
                    # Remove the left side of the tree
                    # left_side_flow = [f for f in workflow['flows']]
                    flow = graph.in_edges(task_id, data=True)
                    # pdb.set_trace()
                    # remove other side
                    pass
                else:
                    flow = [f for f in graph['flows'] if
                            f['target_id'] == task.id and f[
                                'target_port_name'] == 'input data 2']
                    if flow:
                        pdb.set_trace()
                    pass
        return graph


class SparkTranspiler:
    """
    Convert Lemonada workflow representation (JSON) into code to be run in
    Apache Spark.
    """
    DIST_ZIP_FILE = '/tmp/lemonade-lib-python.zip'
    VISITORS = [RemoveTasksWhenMultiplexingVisitor]

    def __init__(self):
        self.operations = {}
        self._assign_operations()
        self.current_task_id = None

    """
    def pre_transpile(self, workflow, graph, out=None, params=None):
        self.graph = graph
        self.params = params if params is not None else {}

        self.using_stdout = out is None
        if self.using_stdout:
            self.out = sys.stdout
        else:
            self.out = out

        self.workflow_json = json.dumps(workflow)
        self.workflow_name = workflow['name']
        self.workflow_id = workflow['id']
        self.workflow_user = workflow.get('user', {})

        self.requires_info = {}

        # dependency = sort_topologically(graph)
        # self.tasks = [all_tasks[item] for sublist in dependency for item in
        #               sublist]

        self.execute_main = False
        for visitor in self.VISITORS:
            self.workflow = visitor().visit(
                self.graph, self.operations, params)
    """

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

    def transpile(self, workflow, graph, params, out=None, job_id=None):
        """ Transpile the tasks from Lemonade's workflow into Spark code """

        using_stdout = out is None
        if using_stdout:
            out = sys.stdout

        ports = {}
        sequential_ports = {}
        for source_id in graph.edge:
            for target_id in graph.edge[source_id]:
                # Nodes accept multiple edges from same source
                for flow in graph.edge[source_id][target_id].values():
                    flow_id = '[{}:{}]'.format(source_id, flow['source_port'], )

                    if flow_id not in sequential_ports:
                        sequential_ports[flow_id] = 'out{}'.format(
                            len(sequential_ports))
# /
                    if source_id not in ports:
                        ports[source_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}
                    if target_id not in ports:
                        ports[target_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}

                    sequence = sequential_ports[flow_id]
                    if sequence not in ports[source_id]['outputs']:
                        ports[source_id]['named_outputs'][
                            flow['source_port_name']] = sequence
                        ports[source_id]['outputs'].append(sequence)
                    if sequence not in ports[target_id]['inputs']:
                        ports[target_id]['named_inputs'][
                            flow['target_port_name']] = sequence
                        ports[target_id]['inputs'].append(sequence)

        env_setup = {'instances': [], 'workflow_name': workflow['name']}

        sorted_tasks_id = nx.topological_sort(graph)
        for i, task_id in enumerate(sorted_tasks_id):
            self.current_task_id = task_id
            task = graph.node[task_id]
            class_name = self.operations[task['operation']['slug']]

            parameters = {}
            for parameter, definition in task['forms'].items():
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

                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']

            # Operation SAVE requires the complete workflow
            if task['operation']['slug'] == 'data-writer':
                parameters['workflow'] = workflow

            # Some temporary variables need to be identified by a sequential
            # number, so it will be stored in this field
            task['order'] = i

            parameters['task'] = task
            parameters['workflow_json'] = json.dumps(workflow)
            parameters['user'] = workflow['user']
            parameters['workflow_id'] = workflow['id']
            parameters['workflow_name'] = workflow['name']
            parameters['operation_id'] = task['operation']['id']
            parameters['task_id'] = task['id']
            parameters['operation_slug'] = task['operation']['slug']
            parameters['job_id'] = job_id
            port = ports.get(task['id'], {})

            instance = class_name(parameters, port.get('inputs', []),
                                  port.get('outputs', []),
                                  port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))
            instance.out_degree = graph.out_degree(task_id)
            env_setup['dependency_controller'] = DependencyController(
                params.get('requires_info', False))

            env_setup['instances'].append(instance)
            env_setup['execute_main'] = params.get('execute_main', False)

        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension])
        template = template_env.get_template("operation.tmpl")
        v = template.render(env_setup)

        if using_stdout:
            out.write(v.encode('utf8'))
        else:
            out.write(v)

    def _assign_operations(self):
        etl_ops = {
            'add-columns': juicer.spark.etl_operation.AddColumns,
            'add-rows': juicer.spark.etl_operation.AddRows,
            'aggregation': juicer.spark.etl_operation.Aggregation,
            'clean-missing': juicer.spark.etl_operation.CleanMissing,
            'difference': juicer.spark.etl_operation.Difference,
            'distinct': juicer.spark.etl_operation.Distinct,
            'drop': juicer.spark.etl_operation.Drop,
            'filter': juicer.spark.etl_operation.Filter,
            # Alias for filter
            'filter-selection': juicer.spark.etl_operation.Filter,
            'intersection': juicer.spark.etl_operation.Intersection,
            'join': juicer.spark.etl_operation.Join,
            # synonym for select
            'projection': juicer.spark.etl_operation.Select,
            # synonym for distinct
            'remove-duplicated-rows': juicer.spark.etl_operation.Distinct,
            'sample': juicer.spark.etl_operation.SampleOrPartition,
            'select': juicer.spark.etl_operation.Select,
            # synonym of intersection'
            'set-intersection': juicer.spark.etl_operation.Intersection,
            'sort': juicer.spark.etl_operation.Sort,
            'split': juicer.spark.etl_operation.RandomSplit,
            'transformation': juicer.spark.etl_operation.Transformation,
        }
        ml_ops = {
            'apply-model': juicer.spark.ml_operation.ApplyModel,
            'classification-model':
                juicer.spark.ml_operation.ClassificationModel,
            'classification-report':
                juicer.spark.ml_operation.ClassificationReport,
            'clustering-model':
                juicer.spark.ml_operation.ClusteringModelOperation,
            'cross-validation':
                juicer.spark.ml_operation.CrossValidationOperation,
            'decision-tree-classifier':
                juicer.spark.ml_operation.DecisionTreeClassifierOperation,
            'evaluate-model': juicer.spark.ml_operation.EvaluateModel,
            'feature-assembler': juicer.spark.ml_operation.FeatureAssembler,
            'feature-indexer': juicer.spark.ml_operation.FeatureIndexer,
            'gaussian-mixture-clustering':
                juicer.spark.ml_operation.GaussianMixtureClusteringOperation,
            'gbt-classifier': juicer.spark.ml_operation.GBTClassifierOperation,

            'k-means-clustering':
                juicer.spark.ml_operation.KMeansClusteringOperation,
            'lda-clustering': juicer.spark.ml_operation.LdaClusteringOperation,
            'naive-bayes-classifier':
                juicer.spark.ml_operation.NaiveBayesClassifierOperation,
            'pearson-correlation':
                juicer.spark.statistic_operation.PearsonCorrelation,
            'perceptron-classifier':
                juicer.spark.ml_operation.PerceptronClassifier,
            'random-forest-classifier':
                juicer.spark.ml_operation.RandomForestClassifierOperation,
            'svm-classification':
                juicer.spark.ml_operation.SvmClassifierOperation,
            'topic-report': juicer.spark.ml_operation.TopicReportOperation,
            'recommendation-model': juicer.spark.ml_operation.RecommendationModel,
            # 'recommendation-model': juicer.spark.ml_operation.CollaborativeOperation,
            'als-recommender':
                juicer.spark.ml_operation.AlternatingLeastSquaresOperation,
            'logistic-model': juicer.spark.ml_operation.LogisticRegressionModel,
            'logistic-regression':
                juicer.spark.ml_operation.LogisticRegressionClassifier,

        }
        data_ops = {
            'change-attribute': juicer.spark.data_operation.ChangeAttribute,
            'data-reader': juicer.spark.data_operation.DataReader,
            'data-writer': juicer.spark.data_operation.Save,
            'external-input':
                juicer.spark.data_operation.ExternalInputOperation,
            'read-csv': juicer.spark.data_operation.ReadCSV,
            'save': juicer.spark.data_operation.Save,

        }
        other_ops = {
            'comment': operation.NoOp,
        }
        geo_ops = {
            'read-shapefile': juicer.spark.geo_operation.ReadShapefile,
            'within': juicer.spark.geo_operation.GeoWithin,
        }
        text_ops = {
            'generate-n-grams':
                juicer.spark.text_operation.GenerateNGramsOperation,
            'remove-stop-words':
                juicer.spark.text_operation.RemoveStopWordsOperation,
            'tokenizer': juicer.spark.text_operation.TokenizerOperation,
            'word-to-vector': juicer.spark.text_operation.WordToVectorOperation
        }
        ws_ops = {
            'multiplexer': juicer.spark.ws_operation.MultiplexerOperation,
            'service-output': juicer.spark.ws_operation.ServiceOutputOperation,

        }
        vis_ops = {
            'publish-as-visualization': juicer.spark.vis_operation.PublishVisOperation,
            'bar-chart': juicer.spark.vis_operation.BarChartOperation,
            'pie-chart': juicer.spark.vis_operation.PieChartOperation,
            'area-chart': juicer.spark.vis_operation.AreaChartOperation,
            'line-chart': juicer.spark.vis_operation.LineChartOperation,
            'table-visualization': juicer.spark.vis_operation.TableVisOperation
        }
        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, ml_ops, other_ops, text_ops,
                    ws_ops, vis_ops]:
            self.operations.update(ops)
