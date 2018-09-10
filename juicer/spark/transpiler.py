# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import hashlib
import sys
import uuid
from collections import OrderedDict

import datetime

import jinja2
import juicer.spark.data_operation as data_operation
import juicer.spark.data_quality_operation as data_quality_operation
import juicer.spark.dm_operation as dm_operation
import juicer.spark.etl_operation as etl_operation
import juicer.spark.feature_operation as feature_operation
import juicer.spark.geo_operation as geo_operation
import juicer.spark.ml_operation as ml_operation
import juicer.spark.statistic_operation as statistic_operation
import juicer.spark.text_operation as text_operation
import juicer.spark.trustworthy_operation as trustworthy_operation
import juicer.spark.vis_operation as vis_operation
import juicer.spark.ws_operation as ws_operation
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

    @staticmethod
    def get_new_task_id():
        return uuid.uuid1()


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


class SparkTranspiler(object):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Apache Spark.
    """
    VISITORS = []
    DATA_SOURCE_OPS = ['data-reader']

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        if slug_to_op_id is None:
            self.slug_to_op_id = {}
        else:
            self.slug_to_op_id = slug_to_op_id
        if port_id_to_port is None:
            self.port_id_to_port = {}
        else:
            self.port_id_to_port = port_id_to_port
        self.operations = {}
        self._assign_operations()
        self.configuration = configuration

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
    def transpile(self, workflow, graph, params, out=None, job_id=None,
                  state=None, deploy=False, export_notebook=False):
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
        self.generate_code(env_setup, graph, job_id, out, params,
                           ports, sorted_tasks_id, state, task_hash,
                           using_stdout, workflow, deploy, export_notebook)

    def get_data_sources(self, workflow):
        return len(
            [t['slug'] in self.DATA_SOURCE_OPS for t in workflow['tasks']]) == 1

    def generate_code(self, env_setup, graph, job_id, out, params, ports,
                      sorted_tasks_id, state, task_hash, using_stdout,
                      workflow, deploy=False, export_notebook=False):

        if deploy:
            # To be able to convert, workflow must obey all these rules:
            # - 1 and exactly 1 data source;
            # - Data source must be defined in Limonero with its attributes in
            # order to define the schema for data input;
            # - For ML models, it is required to have a Save Model operation;
            total_ds = 0
            for task in workflow['tasks']:
                if not task.get('enabled', False):
                    continue
                if task['operation']['slug'] in self.DATA_SOURCE_OPS:
                    total_ds += 1

            if total_ds < 1:
                raise ValueError(_(
                    'Workflow must have at least 1 data source to be deployed.'))
        tasks_ids = sorted_tasks_id
        if deploy:
            tasks_ids = reversed(tasks_ids)

        for i, task_id in enumerate(tasks_ids):
            task = graph.node[task_id]
            class_name = self.operations[task['operation']['slug']]

            parameters = {}
            not_empty_params = [(k, d) for k, d in task['forms'].items() if
                                d['value']]

            task['forms'] = dict(not_empty_params)
            for parameter, definition in task['forms'].items():
                # @FIXME: Fix wrong name of form category
                # (using name instead of category)
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
            if state is None or state.get(task_id) is None:
                parameters['execution_date'] = None
            else:
                v = state.get(task_id, [{}])[0]
                if v:
                    parameters['execution_date'] = v.get('execution_date')
                else:
                    parameters['execution_date'] = None
            parameters['configuration'] = self.configuration
            parameters['operation_id'] = task['operation']['id']
            parameters['task_id'] = task['id']
            parameters['operation_slug'] = task['operation']['slug']
            parameters['job_id'] = job_id
            parameters['display_sample'] = parameters['task']['forms'].get(
                'display_sample', {}).get('value') in (1, '1', True, 'true')
            parameters['display_schema'] = parameters['task']['forms'].get(
                'display_schema', {}).get('value') in (1, '1', True, 'true')
            parameters['user'] = workflow['user']
            parameters['workflow'] = workflow
            parameters['workflow_id'] = workflow['id']
            parameters['workflow_name'] = SparkTranspiler._escape_chars(
                workflow['name'])
            parameters['export_notebook'] = export_notebook
            # parameters['port_id_to_port'] = self.port_id_to_port

            port = ports.get(task['id'], {})

            instance = class_name(parameters, port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))
            instance.out_degree = graph.out_degree(task_id)

            env_setup['instances'].append(instance)
            env_setup['instances_by_task_id'][task['id']] = instance
        env_setup['disabled_tasks'] = workflow['disabled_tasks']
        env_setup['plain'] = params.get('plain', False)
        env_setup['now'] = datetime.datetime.now()
        env_setup['user'] = workflow['user']
        env_setup['execute_main'] = params.get('execute_main', False)
        env_setup['dependency_controller'] = DependencyController(
            params.get('requires_info', False))
        env_setup['transpiler'] = TranspilerUtils()

        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension,
                                                      HandleExceptionExtension,
                                                      'jinja2.ext.do'])
        template_env.globals.update(zip=zip)

        if deploy:
            env_setup['slug_to_op_id'] = self.slug_to_op_id
            # env_setup['slug_to_port_id'] = self.slug_to_port_id
            env_setup['id_mapping'] = {}
            template = template_env.get_template("templates/deploy.tmpl")
            v = template.render(env_setup)
            out.write(v.encode('utf8'))
        elif export_notebook:
            template = template_env.get_template("templates/notebook.tmpl")
            v = template.render(env_setup)
            out.write(v.encode('utf8'))
        else:
            template = template_env.get_template("templates/operation.tmpl")
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
                                                       stand_config[
                                                           'auth_token'],
                                                       job_id, v.encode('utf8'))
                except:
                    pass

    def _assign_operations(self):
        etl_ops = {
            'add-columns': etl_operation.AddColumnsOperation,
            'add-rows': etl_operation.AddRowsOperation,
            'aggregation': etl_operation.AggregationOperation,
            'clean-missing': etl_operation.CleanMissingOperation,
            'difference': etl_operation.DifferenceOperation,
            'distinct': etl_operation.RemoveDuplicatedOperation,
            'drop': etl_operation.DropOperation,
            'execute-python': etl_operation.ExecutePythonOperation,
            'execute-sql': etl_operation.ExecuteSQLOperation,
            'filter': etl_operation.FilterOperation,
            # Alias for filter
            'filter-selection': etl_operation.FilterOperation,
            'intersection': etl_operation.IntersectionOperation,
            'join': etl_operation.JoinOperation,
            # synonym for select
            'projection': etl_operation.SelectOperation,
            # synonym for distinct
            'remove-duplicated-rows': etl_operation.RemoveDuplicatedOperation,
            'replace-value': etl_operation.ReplaceValueOperation,
            'sample': etl_operation.SampleOrPartitionOperation,
            'select': etl_operation.SelectOperation,
            # synonym of intersection'
            'set-intersection': etl_operation.IntersectionOperation,
            'sort': etl_operation.SortOperation,
            'split': etl_operation.SplitOperation,
            'transformation': etl_operation.TransformationOperation,
            'window-transformation':
                etl_operation.WindowTransformationOperation,
        }
        dm_ops = {
            'frequent-item-set': dm_operation.FrequentItemSetOperation,
            'association-rules': dm_operation.AssociationRulesOperation,
            'sequence-mining': dm_operation.SequenceMiningOperation,
        }
        ml_ops = {
            'apply-model': ml_operation.ApplyModelOperation,
            'classification-model': ml_operation.ClassificationModelOperation,
            'classification-report': ml_operation.ClassificationReport,
            'clustering-model': ml_operation.ClusteringModelOperation,
            'cross-validation': ml_operation.CrossValidationOperation,
            'decision-tree-classifier':
                ml_operation.DecisionTreeClassifierOperation,
            'one-vs-rest-classifier': ml_operation.OneVsRestClassifier,
            'evaluate-model': ml_operation.EvaluateModelOperation,
            'feature-assembler': ml_operation.FeatureAssemblerOperation,
            'feature-indexer': ml_operation.FeatureIndexerOperation,
            'gaussian-mixture-clustering':
                ml_operation.GaussianMixtureClusteringOperation,
            'gbt-classifier': ml_operation.GBTClassifierOperation,
            'isotonic-regression': ml_operation.IsotonicRegressionOperation,
            'k-means-clustering': ml_operation.KMeansClusteringOperation,
            'lda-clustering': ml_operation.LdaClusteringOperation,
            'lsh': ml_operation.LSHOperation,
            'naive-bayes-classifier':
                ml_operation.NaiveBayesClassifierOperation,
            'one-hot-encoder': ml_operation.OneHotEncoderOperation,
            'pca': ml_operation.PCAOperation,
            'perceptron-classifier': ml_operation.PerceptronClassifier,
            'random-forest-classifier':
                ml_operation.RandomForestClassifierOperation,
            'svm-classification': ml_operation.SvmClassifierOperation,
            'topic-report': ml_operation.TopicReportOperation,
            'recommendation-model': ml_operation.RecommendationModel,
            'als-recommender': ml_operation.AlternatingLeastSquaresOperation,
            'logistic-regression':
                ml_operation.LogisticRegressionClassifierOperation,
            'linear-regression': ml_operation.LinearRegressionOperation,
            'regression-model': ml_operation.RegressionModelOperation,
            'index-to-string': ml_operation.IndexToStringOperation,
            'random-forest-regressor':
                ml_operation.RandomForestRegressorOperation,
            'gbt-regressor': ml_operation.GBTRegressorOperation,
            'generalized-linear-regressor':
                ml_operation.GeneralizedLinearRegressionOperation,
            'aft-survival-regression':
                ml_operation.AFTSurvivalRegressionOperation,
            'save-model': ml_operation.SaveModelOperation,
            'load-model': ml_operation.LoadModelOperation,
            'voting-classifier': ml_operation.VotingClassifierOperation,
            'outlier-detection': ml_operation.OutlierDetectionOperation,
        }

        data_ops = {
            'change-attribute': data_operation.ChangeAttributeOperation,
            'data-reader': data_operation.DataReaderOperation,
            'data-writer': data_operation.SaveOperation,
            'external-input': data_operation.ExternalInputOperation,
            'read-csv': data_operation.ReadCSVOperation,
            'save': data_operation.SaveOperation,
        }
        data_quality_ops = {
            'entity-matching': data_quality_operation.EntityMatchingOperation,
        }
        statistics_ops = {
            'kaplan-meier-survival':
                statistic_operation.KaplanMeierSurvivalOperation,
            'cox-proportional-hazards':
                statistic_operation.CoxProportionalHazardsOperation
        }
        other_ops = {
            'comment': operation.NoOp,
        }
        geo_ops = {
            'read-shapefile': geo_operation.ReadShapefile,
            'within': geo_operation.GeoWithin,
        }
        text_ops = {
            'generate-n-grams': text_operation.GenerateNGramsOperation,
            'remove-stop-words': text_operation.RemoveStopWordsOperation,
            'tokenizer': text_operation.TokenizerOperation,
            'word-to-vector': text_operation.WordToVectorOperation
        }
        ws_ops = {
            'multiplexer': ws_operation.MultiplexerOperation,
            'service-output': ws_operation.ServiceOutputOperation,

        }
        vis_ops = {
            'publish-as-visualization':
                vis_operation.PublishVisualizationOperation,
            'bar-chart': vis_operation.BarChartOperation,
            'donut-chart': vis_operation.DonutChartOperation,
            'pie-chart': vis_operation.PieChartOperation,
            'area-chart': vis_operation.AreaChartOperation,
            'line-chart': vis_operation.LineChartOperation,
            'table-visualization': vis_operation.TableVisualizationOperation,
            'summary-statistics': vis_operation.SummaryStatisticsOperation,
            'plot-chart': vis_operation.ScatterPlotOperation,
            'scatter-plot': vis_operation.ScatterPlotOperation,
            'map-chart': vis_operation.MapOperation,
            'map': vis_operation.MapOperation
        }
        feature_ops = {
            'bucketizer': feature_operation.BucketizerOperation,
            'quantile-discretizer':
                feature_operation.QuantileDiscretizerOperation,
            'standard-scaler': feature_operation.StandardScalerOperation,
            'max-abs-scaler': feature_operation.MaxAbsScalerOperation,
            'min-max-scaler': feature_operation.MinMaxScalerOperation,

        }
        trustworthy_operations = {
            'fairness-evaluator':
                trustworthy_operation.FairnessEvaluationOperation
        }
        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, ml_ops, other_ops, text_ops,
                    statistics_ops, ws_ops, vis_ops, dm_ops, data_quality_ops,
                    feature_ops, trustworthy_operations]:
            self.operations.update(ops)
