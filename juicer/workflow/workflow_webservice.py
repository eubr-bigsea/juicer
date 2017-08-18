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
        x, y = self.build_webservice_workflow()

        self.workflow_ws['tasks'] = x
        self.graph_ws = y

        # import pdb
        # pdb.set_trace()


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

                    list_predecessors = self.graph_ws.predecessors(workflow_task['id'])
                    list_successors = self.graph_ws.successors(workflow_task['id'])
                    edges_from_node = self.graph_ws.edges(workflow_task['id'])

                    if (self.check_task_operation(workflow_task['id'],
                                                self.webservice_param['inputs'])
                        or self.check_task_operation(workflow_task['id'],
                                                     self.webservice_param['outputs'])
                        or self.check_task_operation(workflow_task['id'],
                                                     self.webservice_param['models'])):

                        # Change operation for a new operation name
                        wf_task_aux = self.change_operations_webservice(workflow_task,
                                                                        self.webservice_lookuptable)

                        self.update_workflow_operation(wf_task_aux)

                        new_workflow_ws.append(wf_task_aux)
                        new_graph_ws.add_node(wf_task_aux.get('id'),
                                              attr_dict=wf_task_aux)

                        # Add arestas
                        for sucessor in list_successors:
                            flow_operation = self.search_task_operation_on_flow(workflow_task['id'], sucessor)
                            new_graph_ws.add_edge(wf_task_aux['id'],sucessor, attr_dict=flow_operation)


                    else:
                        task_operation_retrainmodel = False

                        wf_task_aux = copy.deepcopy(workflow_task)

                        if (workflow_task['operation']['name'] == 'Split' and
                                    task_operation_retrainmodel == False):

                            for sucessor in list_successors:
                                # is webservice?
                                # What webservice operations is it?
                                task_operation = self.search_task_operation_on_workflow(sucessor)

                                if self.is_webservice_operation(sucessor):

                                    # Create new flow
                                    for predecessor in list_predecessors:

                                        if task_operation['operation']['slug'] == 'classification-model':
                                            task_op_predecessor = self.search_task_operation_on_workflow(sucessor)
                                            task_operation['in_degree'] = task_operation['in_degree'] - 1

                                            if self.graph_ws.has_node(workflow_task['id']):
                                                self.graph_ws.remove_node(workflow_task['id'])
                                                removed_tasks.append(workflow_task['id'])

                                        elif task_operation['operation']['slug'] == 'clustering-model':
                                            print "clustering-model"
                                        elif task_operation['operation']['slug'] == 'recommendation-model':
                                            print "recommendation-model"
                                        elif task_operation['operation']['slug'] == 'regression-model':
                                            print "regression-model"
                                        elif task_operation['operation']['slug'] == 'apply-model':
                                            self.graph.remove_edges_from([(workflow_task['id'],sucessor)])
                                            if self.graph.has_node(workflow_task['id']):
                                                self.graph.remove_node(workflow_task['id'])
                                                removed_tasks.append(workflow_task['id'])

                                            flow_operation = self.search_task_operation_on_flow(workflow_task['id'], sucessor)

                                            operation_source = self.search_task_operation_on_workflow(predecessor)
                                            operation_target = self.search_task_operation_on_workflow(sucessor)

                                            ## Add new flow
                                            flow_update = {
                                                'source_port': int(operation_source['operation']['id']),
                                                'source_port_name': operation_source['port_names'][0],
                                                'target_port': int(operation_target['operation']['id']),
                                                'target_port_name': flow_operation['target_port_name'],
                                                'source_id': predecessor,
                                                'target_id':sucessor
                                            }

                                            self.workflow_ws['flows'].append(flow_update)
                                            new_graph_ws.add_edge(predecessor,sucessor, attr_dict=flow_update)

                                            #!@ Remover parents from task['id'] e task_operation
                                            flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                            self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                # Keep the operation
                                # else:

                        else:
                            # Not WS and Not Split operation
                            if (len(edges_from_node)>0 and len(list_successors)>0):
                                # Not is a leaf
                                for sucessor in list_successors:
                                    # is webservice?
                                    if self.is_webservice_operation(sucessor):
                                    # What webservice operations is it?
                                        task_operation = self.search_task_operation_on_workflow(sucessor)

                                        if len(list_predecessors) > 0:
                                            for predecessor in list_predecessors :
                                                # Remover task['id'] from parents of task_operation
                                                # print "Task_operation Not WS", self.graph_ws.edges(task_operation['id'])

                                                if task_operation['operation']['slug'] == 'classification-model' \
                                                        or task_operation['operation']['slug'] == 'read-model':
                                                    ## Criar um novo flow
                                                    ## Remover o flow antigo
                                                    self.graph_ws.remove_edges_from([(workflow_task['id'],sucessor)])
                                                    task_operation['in_degree'] = task_operation['in_degree'] - 1
                                                    if self.graph_ws.has_node(workflow_task['id']):
                                                        self.graph_ws.remove_node(workflow_task['id'])
                                                        # print "Removed task id:: ", workflow_task['id']
                                                        removed_tasks.append(workflow_task['id'])
                                                        for flow in self.workflow['flows']:
                                                            if ((flow.get('source_id') == workflow_task['id'])
                                                                and
                                                                    (flow.get('target_id') == sucessor)):
                                                                # print "Removed task xxxid:: ", workflow_task['id']
                                                                flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                                                self.remove_flow_from_edges(workflow_task['id'],sucessor)

                                                elif task_operation['operation']['slug'] == 'apply-model'\
                                                        or task_operation['operation']['slug'] == 'read-model':
                                                    self.graph_ws.remove_edges_from([(workflow_task['id'],sucessor)])
                                                    task_operation['in_degree'] = task_operation['in_degree'] - 1
                                                    if self.graph_ws.has_node(workflow_task['id']):
                                                        self.graph_ws.remove_node(workflow_task['id'])
                                                        # print "Removed task id:: ", workflow_task['id']
                                                        removed_tasks.append(workflow_task['id'])
                                                        # for flow in self.workflow['flows']:
                                                        #     if ((flow.get('source_id') == workflow_task['id'])
                                                        #         and
                                                        #             (flow.get('target_id') == sucessor)):
                                                                # flow_operation = self.search_task_operation_on_flow(workflow_task['id'],sucessor)
                                                                # self.remove_flow_from_edges(workflow_task['id'],sucessor)
                                                                # print "Removenr parents"
                                        else:

                                            if task_operation['operation']['slug'] == 'classification-model' \
                                                    or task_operation['operation']['slug'] == 'read-model':
                                                old_operation = { 'removed_input' : copy.deepcopy(workflow_task)}

                                                self.update_operation_info(sucessor, old_operation)

                                                if self.graph_ws.has_node(workflow_task['id']):
                                                    removed_tasks.append(workflow_task['id'])
                                                    # print "removing", workflow_task['id']
                                                    self.graph_ws.remove_node(workflow_task['id'])
                                                    self.remove_flow_from_edges(workflow_task['id'], sucessor)

                                    else:
                                        # Sucessor is not a wb operation
                                        inverted_graph = self.graph.reverse(copy=True)
                                        list_dfs_sucessors = self.create_list_from_dictionary(nx.dfs_successors(self.graph_ws, sucessor))
                                        list_dfs_predecessor = self.create_list_from_dictionary(nx.dfs_successors(inverted_graph, sucessor))

                                        if ((len(list_dfs_sucessors) > 0) and
                                                self.has_webservice_in_list(list_dfs_sucessors) and
                                                self.has_webservice_in_list(list_dfs_predecessor) and
                                            len(list_dfs_predecessor) > 0 ):
                                                        # Add arestas
                                            for sucessor in list_successors:
                                                flow_operation = self.search_task_operation_on_flow(workflow_task['id'], sucessor)
                                                new_graph_ws.add_edge(wf_task_aux['id'],sucessor, attr_dict=flow_operation)
                                            pass
                                        else:
                                            self.remove_workflow_task_id_from_graph(workflow_task['id'], removed_tasks)

                            else:
                                # Is a leaf
                                # update flows
                                for predecessor in list_predecessors:

                                    task_operation = self.search_task_operation_on_workflow(predecessor)

                                    # Update out_degree
                                    task_operation['out_degree'] = task_operation['out_degree'] - 1

                                    if task_operation['operation']['slug'] == 'classification-model' \
                                            or task_operation['operation']['slug'] == 'read-model':
                                        # Se task_operation e output?
                                        # Create new operation to represent WS OUTPUT

                                        # Se task_operation e model?
                                        # Apenas remover as conexoes

                                        if self.graph_ws.has_node(workflow_task['id']):
                                            removed_tasks.append(workflow_task['id'])
                                            self.remove_flow_from_edges(predecessor, workflow_task['id'])
                                            self.graph_ws.remove_node(workflow_task['id'])


                                    elif task_operation['operation']['slug'] == 'apply-model' \
                                            or task_operation['operation']['slug'] == 'ws-output':

                                        if self.graph_ws.has_node(workflow_task['id']):
                                            removed_tasks.append(workflow_task['id'])
                                            self.remove_flow_from_edges(predecessor, workflow_task['id'])
                                            self.graph_ws.remove_node(workflow_task['id'])

                        # Add operation only if dictionary not empty
                        # Add workflow_task_aux only if is not empty
                        new_workflow_ws.append(wf_task_aux)
                        new_graph_ws.add_node(wf_task_aux.get('id'),
                                       attr_dict=wf_task_aux)
                        # Check if task is selected as webservice
                        self.remove_tasks_from_workflow(new_workflow_ws, new_graph_ws, removed_tasks)


        return new_workflow_ws, new_graph_ws

    def update_operation_info(self, task_operation, old_operation):
        for element in self.workflow_ws['tasks']:
            if task_operation == element.get('id'):
                element.update(old_operation)

    def update_workflow_operation(self, wf_task_aux):
        for element in self.workflow_ws['tasks']:
            if element.get('id') == wf_task_aux['id']:
                self.workflow_ws['tasks'].remove(element)
                self.workflow_ws['tasks'].append(wf_task_aux)

    def remove_workflow_task_id_from_graph(self, workflow_task_id, removed_tasks):
        if self.graph_ws.has_node(workflow_task_id):
            self.graph_ws.remove_node(workflow_task_id)
            removed_tasks.append(workflow_task_id)

    def search_task_operation_on_flow(self, source_id, target_id):
        for flow in self.workflow_ws['flows']:
            # print flow
            if ((flow.get('source_id') == source_id)
                and
                    (flow.get('target_id') == target_id)):
                return flow

    def remove_flow_from_edges(self, source_id, target_id):

        for flow in self.workflow_ws['flows']:
            if ((flow.get('source_id') == source_id)
                and
                    (flow.get('target_id') == target_id)):
                self.workflow_ws['flows'].remove(flow)

    def remove_tasks_from_workflow(self, new_workflow, new_graph, removed_tasks_list):
        for element in self.workflow_ws['tasks']:
            for parents in element.get('parents'):
                if parents in removed_tasks_list:
                    element.get('parents').remove(parents)

            if element.get('id') in removed_tasks_list:
                self.workflow_ws['tasks'].remove(element)
                if self.graph_ws.has_node(element.get('id')):
                    self.graph_ws.remove_node(element.get('id'))

        for element in new_workflow:
            for parents in element.get('parents'):
                if parents in removed_tasks_list:
                    element.get('parents').remove(parents)

            if element.get('id') in removed_tasks_list:
                new_workflow.remove(element)
                if new_graph.has_node(element.get('id')):
                    new_graph.remove_node(element.get('id'))


    def search_task_operation_on_workflow(self, task_id):
        for task in self.workflow_ws['tasks']:
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

        aux_operation_ws = self.search_task_operation_on_workflow(workflow_task['id'])

        aux_operation =  copy.deepcopy(aux_operation_ws)
        aux_operation['operation']['name'] = dict_lkt[workflow_task['operation']['id']]
        aux_operation['operation']['slug'] = dict_lkt[workflow_task['operation']['id']].lower().replace(" ", "-")
        aux_operation['operation']['id'] = self.table_of_ws_operations[dict_lkt[workflow_task['operation']['id']]]

        old_operation = { 'old_operation' : copy.deepcopy(workflow_task)}
        aux_operation.update(old_operation)

        return aux_operation

    def create_list_from_dictionary(self, dictionary):
        list_of_elements = []
        for key, value in dictionary.iteritems():
            for element in value:
                list_of_elements.append(element)
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