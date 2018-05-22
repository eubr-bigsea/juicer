# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import hashlib
import sys
from collections import OrderedDict

import jinja2
import juicer.spark.data_operation
import juicer.spark.data_quality_operation
import juicer.spark.dm_operation
import juicer.spark.etl_operation
import juicer.spark.feature_operation
import juicer.spark.geo_operation
import juicer.spark.ml_operation
import juicer.spark.statistic_operation
import juicer.spark.text_operation
import juicer.spark.vis_operation
import juicer.spark.ws_operation
import networkx as nx
import os
from juicer import operation
from juicer.service import stand_service
from juicer.util.jinja2_custom import AutoPep8Extension
from juicer.util.template_util import HandleExceptionExtension


class TranspilerUtils(object):
    """ Utilities for using in Jinja2 related to transpiling """

    @staticmethod
    def _get_enabled_tasks_to_execute(instances):
        dependency_controller = DependencyController([])
        result = []
        for instance in TranspilerUtils._get_enabled_tasks(instances):
            task = instance.parameters['task']
            is_satisfied = dependency_controller.is_satisfied(task['id'])
            if instance.must_be_executed(is_satisfied):
                result.append(instance)
        return result

    @staticmethod
    def _get_enabled_tasks(instances):
        return [instance for instance in instances if
                instance.has_code and instance.enabled]

    @staticmethod
    def _get_parent_tasks(instances_map, instance, only_enabled=True):
        if only_enabled:
            dependency_controller = DependencyController([])
            result = []
            for parent_id in instance.parameters['task']['parents']:
                parent = instances_map[parent_id]
                is_satisfied = dependency_controller.is_satisfied(parent_id)
                if is_satisfied and parent.has_code and parent.enabled:
                    method = '{}_{}'.format(
                        parent.parameters['task']['operation']['slug'].replace(
                            '-', '_'), parent.order)
                    result.append((parent_id, method))
            return result
        else:
            return [instances_map[parent_id] for parent_id in
                    instance.parameters['task']['parents']]

    @staticmethod
    def get_ids_and_methods(instances):
        result = OrderedDict()
        for instance in TranspilerUtils._get_enabled_tasks_to_execute(
                instances):
            task = instance.parameters['task']
            result[task['id']] = '{}_{}'.format(
                task['operation']['slug'].replace('-', '_'), instance.order)
        return result

    @staticmethod
    def get_disabled_tasks(instances):
        return [instance for instance in instances if
                not instance.has_code or not instance.enabled]


class DependencyController:
    """ Evaluates if a dependency is met when generating code. """

    def __init__(self, requires):
        self._satisfied = set()
        self.requires = requires

    def satisfied(self, _id):
        self._satisfied.add(_id)

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def is_satisfied(self, _id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


class SparkTranspilerVisitor:
    def __init__(self):
        pass

    def visit(self, workflow, sorted_tasks, operations, params):
        raise NotImplementedError()


class ValidateAttributeReferencesVisitor(SparkTranspilerVisitor):
    def visit(self, workflow, sorted_tasks, operations, params):
        pass


class RemoveTasksWhenMultiplexingVisitor(SparkTranspilerVisitor):
    def visit(self, workflow, sorted_tasks, operations, params):
        return None
        # external_input_op = juicer.spark.ws_operation.MultiplexerOperation
        # for task_id in graph.node:
        #     task = graph.node[task_id]
        #     op = operations.get(task.get('operation').get('slug'))
        #     if op == external_input_op:
        #         # Found root
        #         if params.get('service'):
        #             # Remove the left side of the tree
        #             # left_side_flow = [f for f in workflow['flows']]
        #             flow = graph.in_edges(task_id, data=True)
        #             # pdb.set_trace()
        #             # remove other side
        #             pass
        #         else:
        #             flow = [f for f in graph['flows'] if
        #                     f['target_id'] == task.id and f[
        #                         'target_port_name'] == 'input data 2']
        #             if flow:
        #                 pdb.set_trace()
        #             pass
        # return graph


class SparkTranspiler(object):
    """
    Convert Lemonada workflow representation (JSON) into code to be run in
    Apache Spark.
    """
    VISITORS = [RemoveTasksWhenMultiplexingVisitor]

    def __init__(self, configuration):
        self.operations = {}
        self._assign_operations()
        self.current_task_id = None
        self.configuration = configuration

    def pre_transpile(self, workflow, graph, params=None):
        params = params or {}
        for visitor in self.VISITORS:
            visitor().visit(workflow, nx.topological_sort(graph),
                            self.operations, params)

    @staticmethod
    def _escape_chars(text):
        if isinstance(text, str):
            return text.encode('string-escape').replace('"', '\\"').replace(
                "'", "\\'")
        else:
            return text.encode('unicode-escape').replace('"', '\\"').replace(
                "'", "\\'")

    @staticmethod
    def _gen_port_name(flow, seq):
        name = flow.get('source_port_name', 'data')
        parts = name.split()
        if len(parts) == 1:
            name = name[:5]
        elif name[:3] == 'out':
            name = name[:3]
        else:
            name = ''.join([p[0] for p in parts])
        return '{}{}'.format(name, seq)

    # noinspection SpellCheckingInspection
    def transpile(self, workflow, graph, params, out=None, job_id=None):
        """ Transpile the tasks from Lemonade's workflow into Spark code """

        using_stdout = out is None
        if using_stdout:
            out = sys.stdout

        ports = {}
        sequential_ports = {}
        counter = 0
        for source_id in graph.edge:
            for target_id in graph.edge[source_id]:
                # Nodes accept multiple edges from same source
                for flow in graph.edge[source_id][target_id].values():
                    flow_id = '[{}:{}]'.format(source_id, flow['source_port'], )

                    if flow_id not in sequential_ports:
                        sequential_ports[flow_id] = self._gen_port_name(
                            flow, counter)
                        counter += 1
                    if source_id not in ports:
                        ports[source_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}
                    if target_id not in ports:
                        ports[target_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}

                    sequence = sequential_ports[flow_id]

                    source_port = ports[source_id]
                    if sequence not in source_port['outputs']:
                        source_port['named_outputs'][
                            flow['source_port_name']] = sequence
                        source_port['outputs'].append(sequence)

                    target_port = ports[target_id]
                    if sequence not in target_port['inputs']:
                        flow_name = flow['target_port_name']
                        # Test if multiple inputs connects to a port
                        # because it may have multiplicity MANY
                        if flow_name in target_port['named_inputs']:
                            if not isinstance(
                                    target_port['named_inputs'][flow_name],
                                    list):
                                target_port['named_inputs'][flow_name] = [
                                    target_port['named_inputs'][flow_name],
                                    sequence]
                            else:
                                target_port['named_inputs'][flow_name].append(
                                    sequence)
                        else:
                            target_port['named_inputs'][flow_name] = sequence
                        target_port['inputs'].append(sequence)

        env_setup = {'instances': [], 'instances_by_task_id': {},
                     'workflow_name': workflow['name']}

        sorted_tasks_id = nx.topological_sort(graph)
        task_hash = hashlib.sha1()
        for i, task_id in enumerate(sorted_tasks_id):
            self.current_task_id = task_id
            task = graph.node[task_id]
            class_name = self.operations[task['operation']['slug']]

            parameters = {}
            not_empty_params = [(k, d) for k, d in task['forms'].items() if
                                d['value'] != '' and d['value'] is not None]
            task['forms'] = dict(not_empty_params)
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

                    task_hash.update(unicode(definition['value']).encode(
                        'utf8', errors='ignore'))
                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']
                # escape invalid characters for code generation
                # except JSON (starting with {)
                if definition['value'] is not None and not isinstance(
                        definition['value'], bool):
                    if '"' in definition['value'] or "'" in definition['value']:
                        if definition['value'][0] != '{':
                            definition['value'] = SparkTranspiler._escape_chars(
                                definition['value'])

            # Hash is used in order to avoid re-run task.
            parameters['hash'] = task_hash.hexdigest()

            # Operation SAVE requires the complete workflow
            if task['operation']['slug'] == 'data-writer':
                parameters['workflow'] = workflow

            # Some temporary variables need to be identified by a sequential
            # number, so it will be stored in this field
            parameters['order'] = i

            parameters['task'] = task
            parameters['configuration'] = self.configuration
            parameters['workflow'] = workflow
            parameters['user'] = workflow['user']
            parameters['workflow_id'] = workflow['id']
            parameters['workflow_name'] = SparkTranspiler._escape_chars(
                workflow['name'])
            parameters['operation_id'] = task['operation']['id']
            parameters['task_id'] = task['id']
            parameters['operation_slug'] = task['operation']['slug']
            parameters['job_id'] = job_id
            parameters['display_sample'] = parameters['task']['forms'].get(
                'display_sample', {}).get('value') in (1, '1', True, 'true')
            parameters['display_schema'] = parameters['task']['forms'].get(
                'display_schema', {}).get('value') in (1, '1', True, 'true')
            port = ports.get(task['id'], {})

            instance = class_name(parameters, port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))
            instance.out_degree = graph.out_degree(task_id)

            env_setup['instances'].append(instance)
            env_setup['instances_by_task_id'][task['id']] = instance

        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension,
                                                      HandleExceptionExtension])
        template_env.globals.update(zip=zip)
        template = template_env.get_template("templates/operation.tmpl")

        env_setup['disabled_tasks'] = workflow['disabled_tasks']
        env_setup['plain'] = params.get('plain', False)
        env_setup['execute_main'] = params.get('execute_main', False)
        env_setup['dependency_controller'] = DependencyController(
            params.get('requires_info', False))
        env_setup['transpiler'] = TranspilerUtils()
        v = template.render(env_setup)

        if using_stdout:
            out.write(v.encode('utf8'))
        else:
            out.write(v)
        stand_config = self.configuration.get('juicer', {}).get(
            'services', {}).get('stand')
        if stand_config and job_id:
            # noinspection PyBroadException
            try:
                stand_service.save_job_source_code(stand_config['url'],
                                                   stand_config['auth_token'],
                                                   job_id, v.encode('utf8'))
            except:
                pass

    def _assign_operations(self):
        etl_ops = {
            'add-columns': juicer.spark.etl_operation.AddColumnsOperation,
            'add-rows': juicer.spark.etl_operation.AddRowsOperation,
            'aggregation': juicer.spark.etl_operation.AggregationOperation,
            'clean-missing': juicer.spark.etl_operation.CleanMissingOperation,
            'difference': juicer.spark.etl_operation.DifferenceOperation,
            'distinct': juicer.spark.etl_operation.RemoveDuplicatedOperation,
            'drop': juicer.spark.etl_operation.DropOperation,
            'execute-python': juicer.spark.etl_operation.ExecutePythonOperation,
            'execute-sql': juicer.spark.etl_operation.ExecuteSQLOperation,
            'filter': juicer.spark.etl_operation.FilterOperation,
            # Alias for filter
            'filter-selection': juicer.spark.etl_operation.FilterOperation,
            'intersection': juicer.spark.etl_operation.IntersectionOperation,
            'join': juicer.spark.etl_operation.JoinOperation,
            # synonym for select
            'projection': juicer.spark.etl_operation.SelectOperation,
            # synonym for distinct
            'remove-duplicated-rows':
                juicer.spark.etl_operation.RemoveDuplicatedOperation,
            'replace-value':
                juicer.spark.etl_operation.ReplaceValueOperation,
            'sample': juicer.spark.etl_operation.SampleOrPartitionOperation,
            'select': juicer.spark.etl_operation.SelectOperation,
            # synonym of intersection'
            'set-intersection':
                juicer.spark.etl_operation.IntersectionOperation,
            'sort': juicer.spark.etl_operation.SortOperation,
            'split': juicer.spark.etl_operation.SplitOperation,
            'transformation':
                juicer.spark.etl_operation.TransformationOperation,
            'window-transformation':
                juicer.spark.etl_operation.WindowTransformationOperation,
        }
        dm_ops = {
            'frequent-item-set':
                juicer.spark.dm_operation.FrequentItemSetOperation,
            'association-rules':
                juicer.spark.dm_operation.AssociationRulesOperation,
            'sequence-mining':
                juicer.spark.dm_operation.SequenceMiningOperation,
        }
        ml_ops = {
            'apply-model': juicer.spark.ml_operation.ApplyModelOperation,
            'classification-model':
                juicer.spark.ml_operation.ClassificationModelOperation,
            'classification-report':
                juicer.spark.ml_operation.ClassificationReport,
            'clustering-model':
                juicer.spark.ml_operation.ClusteringModelOperation,
            'cross-validation':
                juicer.spark.ml_operation.CrossValidationOperation,
            'decision-tree-classifier':
                juicer.spark.ml_operation.DecisionTreeClassifierOperation,
            'one-vs-rest-classifier':
                juicer.spark.ml_operation.OneVsRestClassifier,
            'evaluate-model': juicer.spark.ml_operation.EvaluateModelOperation,
            'feature-assembler':
                juicer.spark.ml_operation.FeatureAssemblerOperation,
            'feature-indexer':
                juicer.spark.ml_operation.FeatureIndexerOperation,
            'gaussian-mixture-clustering':
                juicer.spark.ml_operation.GaussianMixtureClusteringOperation,
            'gbt-classifier': juicer.spark.ml_operation.GBTClassifierOperation,
            'isotonic-regression':
                juicer.spark.ml_operation.IsotonicRegressionOperation,
            'k-means-clustering':
                juicer.spark.ml_operation.KMeansClusteringOperation,
            'lda-clustering': juicer.spark.ml_operation.LdaClusteringOperation,
            'lsh': juicer.spark.ml_operation.LSHOperation,
            'naive-bayes-classifier':
                juicer.spark.ml_operation.NaiveBayesClassifierOperation,
            'one-hot-encoder':
                juicer.spark.ml_operation.OneHotEncoderOperation,
            'pca': juicer.spark.ml_operation.PCAOperation,
            'perceptron-classifier':
                juicer.spark.ml_operation.PerceptronClassifier,
            'random-forest-classifier':
                juicer.spark.ml_operation.RandomForestClassifierOperation,
            'svm-classification':
                juicer.spark.ml_operation.SvmClassifierOperation,
            'topic-report': juicer.spark.ml_operation.TopicReportOperation,
            'recommendation-model':
                juicer.spark.ml_operation.RecommendationModel,
            'als-recommender':
                juicer.spark.ml_operation.AlternatingLeastSquaresOperation,
            'logistic-regression':
                juicer.spark.ml_operation.LogisticRegressionClassifierOperation,
            'linear-regression':
                juicer.spark.ml_operation.LinearRegressionOperation,
            'regression-model':
                juicer.spark.ml_operation.RegressionModelOperation,
            'index-to-string': juicer.spark.ml_operation.IndexToStringOperation,
            'random-forest-regressor':
                juicer.spark.ml_operation.RandomForestRegressorOperation,
            'gbt-regressor': juicer.spark.ml_operation.GBTRegressorOperation,
            'generalized-linear-regressor':
                juicer.spark.ml_operation.GeneralizedLinearRegressionOperation,
            'aft-survival-regression':
                juicer.spark.ml_operation.AFTSurvivalRegressionOperation,

            'save-model': juicer.spark.ml_operation.SaveModelOperation,
            'load-model': juicer.spark.ml_operation.LoadModelOperation,
            'voting-classifier':
                juicer.spark.ml_operation.VotingClassifierOperation,
            'outlier-detection':
                juicer.spark.ml_operation.OutlierDetectionOperation,

        }

        data_ops = {
            'change-attribute':
                juicer.spark.data_operation.ChangeAttributeOperation,
            'data-reader': juicer.spark.data_operation.DataReaderOperation,
            'data-writer': juicer.spark.data_operation.SaveOperation,
            'external-input':
                juicer.spark.data_operation.ExternalInputOperation,
            'read-csv': juicer.spark.data_operation.ReadCSVOperation,
            'save': juicer.spark.data_operation.SaveOperation,
        }
        data_quality_ops = {
            'entity-matching':
                juicer.spark.data_quality_operation.EntityMatchingOperation,
        }
        statistics_ops = {
            'kaplan-meier-survival':
                juicer.spark.statistic_operation.KaplanMeierSurvivalOperation,
            'cox-proportional-hazards':
                juicer.spark.statistic_operation.CoxProportionalHazardsOperation
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
            'publish-as-visualization':
                juicer.spark.vis_operation.PublishVisualizationOperation,
            'bar-chart': juicer.spark.vis_operation.BarChartOperation,
            'donut-chart': juicer.spark.vis_operation.DonutChartOperation,
            'pie-chart': juicer.spark.vis_operation.PieChartOperation,
            'area-chart': juicer.spark.vis_operation.AreaChartOperation,
            'line-chart': juicer.spark.vis_operation.LineChartOperation,
            'table-visualization':
                juicer.spark.vis_operation.TableVisualizationOperation,
            'summary-statistics':
                juicer.spark.vis_operation.SummaryStatisticsOperation,
            'plot-chart': juicer.spark.vis_operation.ScatterPlotOperation,
            'scatter-plot': juicer.spark.vis_operation.ScatterPlotOperation,
            'map-chart': juicer.spark.vis_operation.MapOperation,
            'map': juicer.spark.vis_operation.MapOperation
        }
        feature_ops = {
            'bucketizer':
                juicer.spark.feature_operation.BucketizerOperation,
            'quantile-discretizer':
                juicer.spark.feature_operation.QuantileDiscretizerOperation,
            'standard-scaler':
                juicer.spark.feature_operation.StandardScalerOperation,
            'max-abs-scaler':
                juicer.spark.feature_operation.MaxAbsScalerOperation,
            'min-max-scaler':
                juicer.spark.feature_operation.MinMaxScalerOperation,

        }

        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, ml_ops, other_ops, text_ops,
                    statistics_ops,
                    ws_ops, vis_ops, dm_ops, data_quality_ops, feature_ops]:
            self.operations.update(ops)
