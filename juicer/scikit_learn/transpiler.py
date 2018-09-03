# -*- coding: utf-8 -*-
import json
import sys
import hashlib
from collections import OrderedDict
import jinja2
import juicer.scikit_learn.data_operation as io
import juicer.scikit_learn.etl_operation as etl
import juicer.scikit_learn.feature_operation as feature_extraction
import juicer.scikit_learn.geo_operation as geo
import juicer.scikit_learn.associative_operation as associative

import juicer.scikit_learn.model_operation as model
import juicer.scikit_learn.regression_operation as regression
import juicer.scikit_learn.text_operation as text_operations
import juicer.scikit_learn.clustering_operation as clustering
import juicer.scikit_learn.classification_operation as classifiers
import juicer.scikit_learn.vis_operation as vis_operation
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

    @staticmethod
    def is_satisfied(_id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


# noinspection SpellCheckingInspection
class ScikitLearnTranspiler(object):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Scikit-Learn.
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.operations = {}
        self._assign_operations()
        self.execute_main = False

    def transpile(self, workflow, graph, params, out=None, job_id=None):
        """ Transpile the tasks from Lemonade's workflow into COMPSs code """

        ports = {}
        sequential_ports = {}

        for source_id in graph.edge:
            for target_id in graph.edge[source_id]:
                # Nodes accept multiple edges from same source
                for flow in graph.edge[source_id][target_id].values():
                    flow_id = '[{}:{}]'.format(source_id, flow['source_port'], )

                    if flow_id not in sequential_ports:
                        sequential_ports[flow_id] = 'data{}'.format(
                            len(sequential_ports))

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
            task = graph.node[task_id]
            class_name = self.operations[task['operation']['slug']]
            parameters = {}
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
                    task_hash.update(unicode(definition['value']).encode(
                            'utf8', errors='ignore'))
                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']

            # Hash is used in order to avoid re-run task.
            parameters['hash'] = task_hash.hexdigest()

            # Operation SAVE requires the complete workflow
            if task['operation']['name'] == 'SAVE':
                parameters['workflow'] = workflow

            # Some temporary variables need to be identified by a sequential
            # number, so it will be stored in this field
            task['order'] = i

            parameters['task'] = task
            parameters['configuration'] = self.configuration
            parameters['workflow_json'] = json.dumps(workflow)
            parameters['user'] = workflow['user']
            parameters['workflow_id'] = workflow['id']
            parameters['workflow_name'] = workflow['name']
            parameters['operation_id'] = task['operation']['id']
            parameters['task_id'] = task['id']
            parameters['operation_slug'] = task['operation']['slug']
            parameters['job_id'] = job_id
            port = ports.get(task['id'], {})

            instance = class_name(parameters, port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))

            env_setup['dependency_controller'] = DependencyController(
                params.get('requires_info', False))

            env_setup['instances'].append(instance)
            env_setup['instances_by_task_id'][task['id']] = instance
            env_setup['execute_main'] = params.get('execute_main', False)
            env_setup['plain'] = params.get('plain', False)

            dict_msgs = {}
            dict_msgs['task_completed'] = _('Task completed')
            dict_msgs['task_running']   = _('Task running')
            dict_msgs['lemonade_task_completed'] = \
                _('Lemonade task %s completed')
            dict_msgs['lemonade_task_parents'] = \
            _('Parents completed, submitting %s')
            dict_msgs['lemonade_task_started'] = \
                _('Lemonade task %s started')
            dict_msgs['lemonade_task_afterbefore'] = \
                _("Submitting parent task {} "
              "before {}")

            env_setup['dict_msgs'] = dict_msgs
            env_setup['transpiler'] = TranspilerUtils()

        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension,
                                                      HandleExceptionExtension])
        template_env.globals.update(zip=zip)
        template = template_env.get_template("operation.tmpl")
        v = template.render(env_setup)

        if out is None:
            sys.stdout.write(v.encode('utf8'))
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
            except Exception as e:
                print (e)
                pass

    def _assign_operations(self):
        etl_ops = {
            'add-columns': etl.AddColumnsOperation,
            'add-rows': etl.UnionOperation,
            'aggregation': etl.AggregationOperation,  # TODO: agg sem groupby
            'clean-missing': etl.CleanMissingOperation,
            'difference': etl.DifferenceOperation,
            'drop': etl.DropOperation,
            'filter-selection': etl.FilterOperation,
            'join': etl.JoinOperation,
            'projection': etl.SelectOperation,
            'remove-duplicated-rows': etl.DistinctOperation,
            'replace-value': etl.ReplaceValuesOperation,
            'sample': etl.SampleOrPartitionOperation,
            'set-intersection': etl.IntersectionOperation,
            'sort': etl.SortOperation,
            'split': etl.SplitOperation,
            'transformation': etl.TransformationOperation,
            # TODO in 'transformation': test others functions
        }

        data_ops = {
            'data-reader': io.DataReaderOperation,
            'data-writer': io.SaveOperation,
            'save': io.SaveOperation,
            # 'change-attribute': io.ChangeAttributesOperation,
        }

        geo_ops = {
            'read-shapefile': geo.ReadShapefileOperation,
            'stdbscan': geo.STDSCANOperantion,
            'within': geo.GeoWithinOperation,
        }

        ml_ops = {
            # ------ Associative ------#
            'association-rules': associative.AssociationRulesOperation,
            'frequent-item-set': associative.FrequentItemSetOperation,
            'sequence-mining': associative.SequenceMiningOperation,

            # ------ Feature Extraction Operations  ------#
            'feature-assembler': feature_extraction.FeatureAssemblerOperation,
            'min-max-scaler': feature_extraction.MinMaxScalerOperation,
            'max-abs-scaler': feature_extraction.MaxAbsScalerOperation,
            'one-hot-encoder': feature_extraction.OneHotEncoderOperation,
            'pca': feature_extraction.PCAOperation,
            'quantile-discretizer':
                feature_extraction.QuantileDiscretizerOperation,
            'standard-scaler': feature_extraction.StandardScalerOperation,
            # 'feature-indexer': #confirm vector and strings
            #     juicer.compss.feature_operation.FeatureIndexerOperation,
            #
            # ------ Model Operations  ------#
            'apply-model': model.ApplyModelOperation,
            # 'evaluate-model':
            #     juicer.compss.model_operation.EvaluateModelOperation,
            'load-model': model.LoadModel,
            'save-model': model.SaveModel,

            # ------ Clustering      -----#
            'clustering-model': clustering.ClusteringModelOperation,  # OK
            'agglomerative-clustering':
                clustering.AgglomerativeClusteringOperation,
            'dbscan-clustering': clustering.DBSCANClusteringOperation,
            'gaussian-mixture-clustering':
                clustering.GaussianMixtureClusteringOperation,
            'k-means-clustering': clustering.KMeansClusteringOperation,
            'lda-clustering': clustering.LdaClusteringOperation,

            # ------ Classification  -----#
            'classification-model': classifiers.ClassificationModelOperation,
            'decision-tree-classifier':
                classifiers.DecisionTreeClassifierOperation,
            'gbt-classifier': classifiers.GBTClassifierOperation,
            'knn-classifier': classifiers.KNNClassifierOperation,
            'logistic-regression': classifiers.LogisticRegressionOperation,
            'naive-bayes-classifier': classifiers.NaiveBayesClassifierOperation,
            'perceptron-classifier': classifiers.PerceptronClassifierOperation,
            'random-forest-classifier':
                classifiers.RandomForestClassifierOperation,
            'svm-classification': classifiers.SvmClassifierOperation,

            # ------ Regression  -----#
            'regression-model': regression.RegressionModelOperation,  # OK
            'gbt-regressor': regression.GradientBoostingRegressorOperation,
            'huber-regressor': regression.HuberRegressorOperation,
            'isotonic-regression':
                regression.IsotonicRegressionOperation,  # OK - TODO: 1D
            'linear-regression': regression.LinearRegressionOperation,
            'random-forest-regressor':
                regression.RandomForestRegressorOperation,
            'sgd-regressor': regression.SGDRegressorOperation,
        }

        text_ops = {
            'generate-n-grams': text_operations.GenerateNGramsOperation,
            'remove-stop-words': text_operations.RemoveStopWordsOperation,
            'tokenizer': text_operations.TokenizerOperation,
            'word-to-vector': text_operations.WordToVectorOperation
        }

        other_ops = {
            'comment': operation.NoOp,
        }

        ws_ops = {}

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

        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, ml_ops,
                    other_ops, text_ops, ws_ops, vis_ops]:
            self.operations.update(ops)