import logging

import matplotlib.pyplot as plt
import networkx as nx
from juicer.service import tahiti_service
import pdb
import copy

class WorkflowWebService:
    WORKFLOW_DATA_PARAM = 'workflow_data'
    WORKFLOW_GRAPH_PARAM = 'workflow_graph'
    WORKFLOW_GRAPH_SORTED_PARAM = 'workflow_graph_sorted'
    WORKFLOW_PARAM = 'workflow'
    GRAPH_PARAM = 'graph'

    WORKFLOW_GRAPH_SOURCE_ID_PARAM = 'source_id'
    WORKFLOW_GRAPH_TARGET_ID_PARAM = 'target_id'

    WEBSERVICE_OPERATIONS_PARAM = 'webservice_param'
    WEBSERVICE_LOOKUPTABLE_PARAM = 'webservice_lookuptable'

    log = logging.getLogger(__name__)

    table_of_ws_operations = {
        'Read Model': 9000,
        'WS Input': 9001,
        'WS Output': 9002,
        'WS Visualization': 9010
    }

    def __init__(self, workflow_data, workflow_graph, params_ws, dict_lkt, config):

        self.config = config

        self.webservice_param = params_ws

        self.webservice_lookuptable = dict_lkt

        # Initialize
        self.graph = nx.MultiDiGraph()
        self.graph_ws = nx.MultiDiGraph()

        # Workflow dictionary
        self.workflow = workflow_data
        self.workflow_ws = copy.deepcopy(workflow_data)

        # Update graph
        self.graph = workflow_graph

        self.graph_ws = copy.deepcopy(workflow_graph)

        # Need update 3 objects
        # 1#: workflow['tasks']
        # 2#: workflow['flows']
        # 3#: workflow_graph

        # call function to change the workflow
        self.build_webservice_workflow()


        # webservice_workflow, webservice_graph = self.builds_ws_workflow()

        # self.workflow['tasks'] = webservice_workflow

        # self.workflow['flows'] =

        # self.graph_webservice = webservice_graph

    def build_webservice_workflow(self):
        """ Builds a workflow service from original workflow """
        # Topological sorted tasks according to their dependencies
        sorted_tasks = self.get_topological_sorted_tasks()

        removed_tasks = []
        new_workflow_ws = []
        # Initialize
        new_graph_ws = nx.MultiDiGraph()

        for task_id in sorted_tasks:
            for workflow_task in self.workflow['tasks']:

                if task_id == workflow_task['id']:
                    print "Condiditon:", workflow_task['operation']['slug']
                    if (self.check_task_operation(workflow_task['id'],
                                                self.webservice_param['inputs'])
                        or self.check_task_operation(workflow_task['id'],
                                                     self.webservice_param['outputs'])
                        or self.check_task_operation(workflow_task['id'],
                                                     self.webservice_param['models'])):
                        # Change operation for a new operation name
                        wf_task_aux = self.change_operations_webservice(workflow_task,
                                                                        self.webservice_lookuptable)

                    else:
                        task_operation_retrainmodel = False

                        wf_task_aux = copy.deepcopy(workflow_task)

                        list_predecessors = self.graph.predecessors(workflow_task['id'])
                        list_successors = self.graph.successors(workflow_task['id'])
                        edges_from_node = self.graph.edges(workflow_task['id'])

                        if (workflow_task['operation']['name'] == 'Split' and
                                    task_operation_retrainmodel == False):
                            print "remover edges from split"



                            for sucessor in list_successors:
                                # is webservice?
                                if self.is_webservice_operation(sucessor):
                                    # What webservice operations is it?
                                    task_operation = self.search_task_operation_on_workflow(sucessor)

                                    for predecessor in list_predecessors:

                                        if task_operation['operation']['slug'] == 'classification-model':
                                            print "classification-model"
                                            task_op_predecessor = self.search_task_operation_on_workflow(sucessor)
                                            ## Criar um novo flow
                                            ## Remover o flow antigo



                                            self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                            if self.graph.has_node(workflow_task['id']):
                                                self.graph.remove_node(workflow_task['id'])
                                                removed_tasks.append(workflow_task['id'])

                                                for flow in self.workflow['flows']:
                                                    if ((flow.get('source_id') == workflow_task['id'])
                                                        and
                                                            (flow.get('target_id') == sucessor)):
                                                        # flow_operation = self.search_task_operation_on_flow(task['id'],sucessor)
                                                        # self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                                        print "#!@ Remover parents from task['id'] e task_operation"
                                        elif task_operation['operation']['slug'] == 'clustering-model':
                                            print "clustering-model"
                                        elif task_operation['operation']['slug'] == 'recommendation-model':
                                            print "recommendation-model"
                                        elif task_operation['operation']['slug'] == 'regression-model':
                                            print "regression-model"
                                        # elif task_operation['operation']['slug'] == 'apply-model':
                                        #     print "apply-model"
                                        #
                                        #     self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                        #     if self.graph.has_node(workflow_task['id']):
                                        #         self.graph.remove_node(workflow_task['id'])
                                        #         removed_tasks.append(workflow_task['id'])
                                        #
                                        #     flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                        #
                                        #     operation_source = self.search_task_operation_on_workflow(predecessor)
                                        #     operation_target = self.search_task_operation_on_workflow(sucessor)
                                        #
                                        #     ## Add new flow
                                        #     flow_update = {
                                        #         'source_port': int(operation_source['operation']['id']),
                                        #         'source_port_name': operation_source['port_names'][0],
                                        #         'target_port': int(operation_target['operation']['id']),
                                        #         'target_port_name': flow_operation['target_port_name'],
                                        #         'source_id': predecessor,
                                        #         'target_id':sucessor
                                        #     }
                                        #
                                        #     self.workflow['flows'].append(flow_update)
                                        #     self.graph.add_edge(predecessor,sucessor, attr_dict=flow_update)
                                        #
                                        #     #!@ Remover parents from task['id'] e task_operation
                                        #     flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                        #     self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                        #
                                        #
                                        # elif task_operation['operation']['slug'] == 'naive-bayes-classifier':
                                        #     print "naive-bayes-classifier"
                                        #     task_op_predecessor = self.search_task_operation_on_workflow(sucessor)
                                        #     self.graph.add_edge(predecessor,sucessor)
                                        #     #
                                        #     self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                        #     if self.graph.has_node(workflow_task['id']):
                                        #         self.graph.remove_node(workflow_task['id'])
                                        #         flow_operation = self.search_task_operation_on_flow(self, workflow_task['id'],sucessor)
                                        #
                                        #         #!@ Remover parents from task['id'] e task_operation


                                else:
                                    print "!!! keep the operation "

                        else:
                            print "Not WS, Not Split"

                            if (len(edges_from_node)>0 and len(list_successors)>0):
                                # Not is a leaf
                                # task_aux = self.change_operations_not_webservice(task, self.webservice_lookuptable)
                                for sucessor in list_successors:
                                    # is webservice?
                                    if self.is_webservice_operation(sucessor):
                                        # What webservice operations is it?
                                        task_operation = self.search_task_operation_on_workflow(sucessor)

                                        for predecessor in list_predecessors:
                                            # Remover task['id'] from parents of task_operation
                                            print "Task_operation Not WS", self.graph.edges(task_operation['id'])

                                            if task_operation['operation']['slug'] == 'classification-model':
                                                print "classification-model"
                                                ## Criar um novo flow
                                                ## Remover o flow antigo
                                                self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                                if self.graph.has_node(workflow_task['id']):
                                                    self.graph.remove_node(workflow_task['id'])
                                                    print "Removed task id:: ", workflow_task['id']
                                                    removed_tasks.append(workflow_task['id'])
                                                    for flow in self.workflow['flows']:
                                                        if ((flow.get('source_id') == workflow_task['id'])
                                                            and
                                                                (flow.get('target_id') == sucessor)):
                                                            print "Removed task xxxid:: ", workflow_task['id']
                                                            flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                                            self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                            elif task_operation['operation']['slug'] == 'classification-model':
                                                print "apply-model"
                                                ## Criar um novo flow
                                                ## Remover o flow antigo
                                                self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                                if self.graph.has_node(workflow_task['id']):
                                                    self.graph.remove_node(workflow_task['id'])
                                                    print "Removed task id:: ", workflow_task['id']
                                                    removed_tasks.append(task['id'])
                                                    for flow in self.workflow['flows']:
                                                        if ((flow.get('source_id') == workflow_task['id'])
                                                            and
                                                                (flow.get('target_id') == sucessor)):
                                                            # flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                                            # self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                                            print "Removenr parents"

                            else:
                                print "# Remove edges to task['id']", workflow_task['id']
                                # removed_tasks.append(workflow_task['id'])

                                for predecessor in list_predecessors:

                                    task_operation = self.search_task_operation_on_workflow(predecessor)

                                    if task_operation['operation']['slug'] == 'classification-model':
                                        print "classification-model"
                                        self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                        if self.graph.has_node(workflow_task['id']):
                                            self.graph.remove_node(workflow_task['id'])
                                            removed_tasks.append(workflow_task['id'])
                                            for flow in self.workflow['flows']:
                                                # print type(flow), flow
                                                if ((flow.get('source_id') == workflow_task['id'])
                                                    and
                                                        (flow.get('target_id') == sucessor)):
                                                    # flow_operation = self.search_task_operation_on_flow(task['id'],sucessor)
                                                    # self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                                    print ">>>"

                                    elif task_operation['operation']['slug'] == 'apply-model':
                                        print "apply-model"
                                        self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                        if self.graph.has_node(workflow_task['id']):
                                            self.graph.remove_node(workflow_task['id'])
                                            removed_tasks.append(workflow_task['id'])


                    # Add operation only if dictionary not empty
                    # print "new_operation_result", task_aux
                    new_workflow_ws.append(wf_task_aux)
                    new_graph_ws.add_node(wf_task_aux.get('id'),
                                       attr_dict=wf_task_aux)

        # Check if task is selected as webservice
        self.remove_tasks_from_workflow(new_workflow_ws, new_graph_ws, removed_tasks)

        # print self.workflow_ws
        print len(self.graph_ws.nodes())
        print "Removed nodes", removed_tasks

    def remove_tasks_from_workflow(self, new_workflow, new_graph, removed_tasks_list):
        # print " # Removing tasks from workflow and graph"
        for element in self.workflow_ws['tasks']:
            for parents in element.get('parents'):
                if parents in removed_tasks_list:
                    element.get('parents').remove(parents)

            if element.get('id') in removed_tasks_list:
                self.workflow_ws['tasks'].remove(element)
                try:
                    self.graph_ws.remove_node(element.get('id'))
                except:
                    pass

        # return new_workflow, new_graph

    def search_task_operation_on_workflow(self, task_id):
        for task in self.workflow['tasks']:
            if task['id'] == task_id:
                return task

    def is_webservice_operation(self, task):
        if (self.check_task_operation(task, self.webservice_param['inputs'])
            or
                self.check_task_operation(task, self.webservice_param['outputs'])
            or
                self.check_task_operation(task, self.webservice_param['models'])):
            return True
        else:
            return False

    def change_operations_webservice(self, workflow_task, dict_lkt):

        aux_operation =  copy.deepcopy(workflow_task)
        parents_operations = workflow_task['parents']

        aux_operation['operation']['name'] = dict_lkt[workflow_task['operation']['id']]
        aux_operation['operation']['slug'] = dict_lkt[workflow_task['operation']['id']].lower().replace(" ", "-")
        aux_operation['operation']['id'] = self.table_of_ws_operations[dict_lkt[workflow_task['operation']['id']]]

        old_operation = { 'old_operation' : copy.deepcopy(workflow_task)}
        aux_operation.update(old_operation)

        return aux_operation

    def create_list_from_dictionary(self, dictionary):
        list_of_elements = []
        for key, value in dictionary.iteritems():
            # list_of_elements.append(key)
            for element in value:
                list_of_elements.append(element)
        # return list(set(list_of_elements))
        return list_of_elements

    def check_task_operation(self, workflow_task_id, list_params_ws):

        for webservice_operations in list_params_ws:
            if (workflow_task_id == webservice_operations['operation_id']):
                return True
        return False

    def has_webservice(self, parents_list):
        is_web_service_successors = False
        for parents in parents_list:
            if (self.check_task_operation(parents, self.webservice_param['inputs'])
                or
                    self.check_task_operation(parents, self.webservice_param['outputs'])
                or
                    self.check_task_operation(parents, self.webservice_param['models'])):
                return True

        return is_web_service_successors


    def has_webservice_in_list(self, list_dfs):
        is_web_service_successors = False
        for element in list_dfs:
            if (self.check_task_operation(element, self.webservice_param['inputs'])
                or
                    self.check_task_operation(element, self.webservice_param['outputs'])
                or
                    self.check_task_operation(element, self.webservice_param['models'])):
                return True
            else:
                is_web_service_successors = False
        return is_web_service_successors

    def get_topological_sorted_tasks(self):

        """ Create the tasks Graph and perform topological sorting

            A topological sort is a nonunique permutation of the nodes
            such that an edge from u to v implies that u appears before
            v in the topological sort order.

            :return: Return a list of nodes in topological sort order.
        """
        # First, map the tasks IDs to their original position
        tasks_position = {}

        for count_position, task in enumerate(self.workflow['tasks']):
            tasks_position[task['id']] = count_position

        sorted_tasks_id = nx.topological_sort(self.graph, reverse=False)

        return sorted_tasks_id


def builds_ws_workflow(self):
    """ Builds a workflow service """