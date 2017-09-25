# -*- coding: utf-8 -*-
import json
import sys

import jinja2
import juicer.compss.data_operation
import juicer.compss.etl_operation
import juicer.compss.geo_operation
import juicer.compss.graph_operation
import juicer.compss.ml_operation
import juicer.compss.text_operation
import networkx as nx
import os
from juicer import operation
from juicer.service import stand_service
from juicer.util.jinja2_custom import AutoPep8Extension
from juicer.util.template_util import HandleExceptionExtension


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
class COMPSsTranspiler(object):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    COMPSs.
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.operations = {}
        self._assign_operations()

        # self.graph = graph
        # self.params = params if params is not None else {}
        #
        # self.using_stdout = out is None
        # if self.using_stdout:
        #     self.out = sys.stdout
        # else:
        #     self.out = out
        #
        # self.job_id = job_id
        # workflow_json = json.dumps(workflow)
        # workflow_name = workflow['name']
        # workflow_id = workflow['id']
        # workflow_user = workflow.get('user', {})

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

        env_setup = {'instances': [], 'workflow_name': workflow['name']}

        sorted_tasks_id = nx.topological_sort(graph)

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

                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']

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
            env_setup['execute_main'] = self.execute_main

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
            except:
                pass

    def _assign_operations(self):
        etl_ops = {
            'add-columns': juicer.compss.etl_operation.AddColumnsOperation,
            'add-rows': juicer.compss.etl_operation.UnionOperation,
            'aggregation': juicer.compss.etl_operation.AggregationOperation,
            'clean-missing': juicer.compss.etl_operation.CleanMissingOperation,
            'difference': juicer.compss.etl_operation.DifferenceOperation,

            'drop': juicer.compss.etl_operation.DropOperation,
            'filter-selection': juicer.compss.etl_operation.FilterOperation,
            'join': juicer.compss.etl_operation.JoinOperation,

            'projection': juicer.compss.etl_operation.SelectOperation,
            'remove-duplicated-rows':
                juicer.compss.etl_operation.DistinctOperation,
            'replace-value': juicer.compss.etl_operation.ReplaceValuesOperation,

            'sample': juicer.compss.etl_operation.SampleOrPartition,
            'set-intersection': juicer.compss.etl_operation.Intersection,
            'sort': juicer.compss.etl_operation.SortOperation,
            'split': juicer.compss.etl_operation.SplitOperation,
            'transformation':
                juicer.compss.etl_operation.TransformationOperation,

        }

        data_ops = {
            'data-reader': juicer.compss.data_operation.DataReaderOperation,
            'data-writer': juicer.compss.data_operation.SaveOperation,
            'save': juicer.compss.data_operation.SaveOperation,
            'balance-data':
                juicer.compss.data_operation.WorkloadBalancerOperation,
            'change-attributes':
                juicer.compss.data_operation.ChangeAttributesOperation,
        }

        geo_ops = {
            # 'read-shapefile':
            # juicer.compss.geo_operation.ReadShapefileOperation,
            # 'within':
            # juicer.compss.geo_operation.GeoWithinOperation,
        }

        graph_ops = {
            'page-rank': juicer.compss.graph_operation.PageRankOperation,
        }

        ml_ops = {
            # ------ Associative ------#
            'frequent-item-set': juicer.compss.ml_operation.AprioriOperation,
            'association-rules':
                juicer.compss.ml_operation.AssociationRulesOperation,

            # ------ Feature Extraction Operations  ------#

            'feature-assembler':
                juicer.compss.ml_operation.FeatureAssemblerOperation,
            'feature-indexer':
                juicer.compss.ml_operation.FeatureIndexerOperation,

            # ------ Model Operations  ------#
            'apply-model': juicer.compss.ml_operation.ApplyModel,

            # ------ Clustering      -----#
            'clustering-model':
                juicer.compss.ml_operation.ClusteringModelOperation,
            'k-means-clustering':
                juicer.compss.ml_operation.KMeansClusteringOperation,

            # ------ Classification  -----#
            'classification-model':
                juicer.compss.ml_operation.ClassificationModelOperation,

            'knn-classifier': juicer.compss.ml_operation.KNNClassifierOperation,
            'logistic-regression':
                juicer.compss.ml_operation.LogisticRegressionOperation,
            'naive-bayes-classifier':
                juicer.compss.ml_operation.NaiveBayesClassifierOperation,
            'svm-classification':
                juicer.compss.ml_operation.SvmClassifierOperation,

            # ------ Evaluation  -----#
            'evaluate-model': juicer.compss.ml_operation.EvaluateModelOperation,

            # ------ Regression  -----#
            'regression-model':
                juicer.compss.ml_operation.RegressionModelOperation,
            'linear-regression':
                juicer.compss.ml_operation.LinearRegressionOperation,

        }

        text_ops = {
            'remove-stop-words':
                juicer.compss.text_operation.RemoveStopWordsOperation,
            'tokenizer': juicer.compss.text_operation.TokenizerOperation,
            'word-to-vector': juicer.compss.text_operation.WordToVectorOperation
        }

        other_ops = {
            'comment': operation.NoOp,
        }

        ws_ops = {}
        vis_ops = {}

        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, graph_ops, ml_ops,
                    other_ops, text_ops, ws_ops, vis_ops]:
            self.operations.update(ops)
