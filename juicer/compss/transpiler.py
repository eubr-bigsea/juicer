# -*- coding: utf-8 -*-
import json
import pdb
import sys
import jinja2
import os
import networkx as nx
from juicer.util.jinja2_custom import AutoPep8Extension

from   juicer import operation
import juicer.compss.data_operation
import juicer.compss.etl_operation
import juicer.compss.ml_operation

class DependencyController:
    """ Evaluates if a dependency is met when generating code. """

    def __init__(self, requires):
        self._satisfied = set()
        self.requires = requires

    def satisfied(self, _id):
        self._satisfied.add(_id)

    def is_satisfied(self, _id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


class COMPSsTranspilerVisitor:
    def __init__(self):
        pass

    def visit(self, workflow, operations, params):
        raise NotImplementedError()

class RemoveTasksWhenMultiplexingVisitor(COMPSsTranspilerVisitor):
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
                    #pdb.set_trace()
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

class COMPSsTranspiler:
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    COMPSs.
    """
    #DIST_ZIP_FILE = '/tmp/lemonade-lib-python.zip'
    VISITORS = [RemoveTasksWhenMultiplexingVisitor]

    def __init__(self, workflow, graph, out=None, params=None):
        self.operations = {}
        self._assign_operations()
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

        self.execute_main = False
        for visitor in self.VISITORS:
            self.workflow = visitor().visit(self.graph, self.operations, params)

    def transpile(self):
        """ Transpile the tasks from Lemonade's workflow into COMPSs code """

        ports = {}
        sequential_ports = {}

        for source_id in self.graph.edge:
            for target_id in self.graph.edge[source_id]:
                # Nodes accept multiple edges from same source
                for flow in self.graph.edge[source_id][target_id].values():
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
                    if sequence not in ports[source_id]['outputs']:
                        ports[source_id]['named_outputs'][
                            flow['source_port_name']] = sequence
                        ports[source_id]['outputs'].append(sequence)
                    if sequence not in ports[target_id]['inputs']:
                        ports[target_id]['named_inputs'][
                            flow['target_port_name']] = sequence
                        ports[target_id]['inputs'].append(sequence)

        env_setup = {'instances': [], 'workflow_name': self.workflow_name}

        id_model= -1;
        sorted_tasks_id = nx.topological_sort(self.graph)
        for i, task_id in enumerate(sorted_tasks_id):
            task = self.graph.node[task_id]
            # if( 'k-means-clustering' == task['operation']['slug']):
            #     print "oi"
            #     cluster= self.graph.node[task_id]['id']
            #     print cluster
            #     id_model =  self.graph.edge[self.graph.node[task_id]['id']]
            #     print "end"
            # if id_model == self.graph.node[task_id]['id']:
            #     print "aqui"
            #     print self.graph.node[cluster]
            class_name = self.operations[task['operation']['slug']]
            parameters = {}
            for parameter, definition in task['forms'].iteritems():
                # @FIXME: Fix wrong name of form category
                # (using name instead of category)
                # print definition.get('category')
                # raw_input()
                cat = definition.get('category','execution').lower()  # FIXME!!!
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
                parameters['workflow'] = self.workflow

            # Some temporary variables need to be identified by a sequential
            # number, so it will be stored in this field
            task['order'] = i

            parameters['task'] = task
            parameters['workflow_json'] = self.workflow_json
            parameters['user'] = self.workflow_user
            parameters['workflow_id'] = self.workflow_id
            port = ports.get(task['id'], {})

            #print task
            #print port

            instance = class_name(parameters, port.get('named_inputs', {}), port.get('named_outputs', {}))

            env_setup['dependency_controller'] = DependencyController(self.requires_info)
            env_setup['instances'].append(instance)
            env_setup['execute_main'] = self.execute_main

        template_loader = jinja2.FileSystemLoader( searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(loader=template_loader, extensions=[AutoPep8Extension])
        template = template_env.get_template("operation.tmpl")
        v = template.render(env_setup)
        if self.using_stdout:
            self.out.write(v.encode('utf8'))
        else:
            self.out.write(v)

    def _assign_operations(self):
        etl_ops = {
            'add-columns':      juicer.compss.etl_operation.AddColumns,
            'add-rows':         juicer.compss.etl_operation.AddRows,
            'difference':       juicer.compss.etl_operation.Difference,
            'distinct':         juicer.compss.etl_operation.Distinct,
            'drop':             juicer.compss.etl_operation.Drop,
            'intersection':     juicer.compss.etl_operation.Intersection,
            'select':           juicer.compss.etl_operation.Select,
            # synonym of intersection'
            'set-intersection': juicer.compss.etl_operation.Intersection,
            #'filter': juicer.compss.etl_operation.Filter, #
            # Alias for filter
            #'filter-selection': juicer.spark.etl_operation.Filter,#
            'intersection':     juicer.compss.etl_operation.Intersection,
            # synonym for select
            'projection':       juicer.compss.etl_operation.Select,
            # synonym for distinct
            'remove-duplicated-rows': juicer.compss.etl_operation.Distinct,
            #'sample': juicer.spark.etl_operation.SampleOrPartition,#


        }

        data_ops = {
            'data-reader': juicer.compss.data_operation.DataReader,
            'data-writer': juicer.compss.data_operation.Save,
            'save':        juicer.compss.data_operation.Save,
        }

        geo_ops = {}

        ml_ops = {
            'apply-model':              juicer.compss.ml_operation.ApplyModel,
            'k-means-clustering':       juicer.compss.ml_operation.KMeansClusteringOperation,
            'clustering-model':         juicer.compss.ml_operation.ClusteringModelOperation,
            'svm-classification':       juicer.compss.ml_operation.SvmClassifierOperation,
            #   'naive-bayes-classifier':   juicer.compss.ml_operation.NaiveBayesClassifierOperation,  #####################
            'classification-model':     juicer.compss.ml_operation.ClassificationModelOperation,            #####################

        }

        text_ops = {

        }

        other_ops = {
            'comment': operation.NoOp,
        }

        ws_ops  = {}
        vis_ops = {}

        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, ml_ops, other_ops, text_ops,
                    ws_ops, vis_ops]:
            self.operations.update(ops)
