# coding=utf-8

import collections
import logging
import re
import datetime
from gettext import gettext

import networkx as nx
from juicer import privaaas
from juicer.service import tahiti_service, limonero_service


class Workflow(object):
    """
        - Set and get Create a graph
        - Identify tasks and flows
        - Set and get workflow
        - Add edges between tasks (source_id and targed_id)

    """
    WORKFLOW_DATA_PARAM = 'workflow_data'
    WORKFLOW_GRAPH_PARAM = 'workflow_graph'
    WORKFLOW_GRAPH_SORTED_PARAM = 'workflow_graph_sorted'
    WORKFLOW_PARAM = 'workflow'
    GRAPH_PARAM = 'graph'

    WORKFLOW_GRAPH_SOURCE_ID_PARAM = 'source_id'
    WORKFLOW_GRAPH_TARGET_ID_PARAM = 'target_id'

    log = logging.getLogger(__name__)

    def __init__(self, workflow_data, config, query_operations=None,
                 query_data_sources=None):
        """
        Constructor.
        :param workflow_data: Workflow dictionary
        :param config: Execution configuration
        :param query_operations: how query operations, useful for testing
        :param query_data_sources: how query data sources, useful for testing
        """
        self.config = config
        self.query_operations = query_operations
        self.query_data_sources = query_data_sources

        # Initialize
        self.graph = nx.MultiDiGraph()

        # Tasks disabled do not execute
        self.disabled_tasks = {}

        # Workflow dictionary
        self.workflow = workflow_data
        self.workflow['data_source_cache'] = {}
        self.workflow['disabled_tasks'] = self.disabled_tasks

        # Construct graph
        self._build_initial_workflow_graph()

        # Topological sorted tasks according to their dependencies
        self.sorted_tasks = []

        # Spark or COMPSs
        self.platform = workflow_data.get('platform', {})

        if self.platform['slug'] == 'spark':
            self._build_privacy_restrictions()

        # Verify null edges to topological_sorted_tasks
        if self._is_there_null_target_id_tasks() \
                and self._is_there_null_source_id_tasks():
            self.sorted_tasks = self.get_topological_sorted_tasks()
        else:
            raise AttributeError(
                _("Port '{}/{}' must be informed for operation{}").format(
                    self.WORKFLOW_GRAPH_SOURCE_ID_PARAM,
                    self.WORKFLOW_GRAPH_TARGET_ID_PARAM,
                    self.__class__))

    def _build_privacy_restrictions(self):
        if 'juicer' not in self.config or \
                        'services' not in self.config['juicer']:
            return
        limonero_config = self.config['juicer']['services']['limonero']
        data_sources = []
        if self.workflow['platform']['slug'] != 'spark':
            return
        for t in self.workflow['tasks']:
            if t['operation'].get('slug') == 'data-reader':
                if self.query_data_sources:
                    ds = next(self.query_data_sources())
                else:
                    ds = limonero_service.get_data_source_info(
                        limonero_config['url'],
                        str(limonero_config['auth_token']),
                        t['forms']['data_source']['value'])
                data_sources.append(ds)

        privacy_info = {}
        attribute_group_set = collections.defaultdict(list)
        data_source_cache = {}
        for ds in data_sources:
            data_source_cache[ds['id']] = ds
            attrs = []
            privacy_info[ds['id']] = {'attributes': attrs}
            for attr in ds['attributes']:
                privacy = attr.get('attribute_privacy', {}) or {}
                attribute_privacy_group_id = privacy.get(
                    'attribute_privacy_group_id')
                privacy_config = {
                    'id': attr['id'],
                    'name': attr['name'],
                    'type': attr['type'],
                    'details': privacy.get('hierarchy'),
                    'privacy_type': privacy.get('privacy_type'),
                    'anonymization_technique': privacy.get(
                        'anonymization_technique'),
                    'attribute_privacy_group_id': attribute_privacy_group_id
                }
                attrs.append(privacy_config)
                if attribute_privacy_group_id:
                    attribute_group_set[attribute_privacy_group_id].append(
                        privacy_config)
                    # print('#' * 40)
                    # print(attr.get('name'), attr.get('type'))
                    # print(privacy.get('privacy_type'),
                    #       privacy.get('anonymization_technique'),
                    #       privacy.get('attribute_privacy_group_id'))

        def sort_attr_privacy(a):
            return privaaas.ANONYMIZATION_TECHNIQUES[a.get(
                'anonymization_technique', 'NO_TECHNIQUE')]

        for attributes in list(attribute_group_set.values()):
            more_restrictive = sorted(
                attributes, key=sort_attr_privacy, reverse=True)[0]
            # print(json.dumps(more_restrictive[0], indent=4))
            # Copy all privacy config from more restrictive one
            for attribute in attributes:
                attribute.update(more_restrictive)

        self.workflow['data_source_cache'] = data_source_cache
        self.workflow['privacy_restrictions'] = privacy_info

    def _build_initial_workflow_graph(self):
        """ Builds a graph with the tasks """

        operations_tahiti = {op['id']: op for op 
                in self._get_operations(self.workflow['id'])}
        # Querying all operations from tahiti one time
        task_map = {}

        all_task_names = []

        for task in self.workflow['tasks']:
            if not task.get('name') in all_task_names:
                all_task_names.append(task.get('name'))
            else:
                raise ValueError(gettext('Task names must be unique.'))

            if task.get('enabled', True) and task.get(
                    'environment', 'DESIGN') == 'DESIGN':
                operation = operations_tahiti.get(task['operation']['id'])
                form_fields = {}
                for form in operation['forms']:
                    for field in form['fields']:
                        form_fields[field['name']] = form['category']

                task_map[task['id']] = {'task': task, 'operation': operation}
                if operation:
                    # Slug information is required in order to select which
                    # operation will be executed
                    task['operation']['slug'] = operation['slug']
                    task['operation']['name'] = operation['name']
                    task['operation']['ports'] = dict(
                        [
                            (
                                port['id'],
                                {
                                    'id': port['id'],
                                    'type': port['type'],
                                    'slug': port['slug'],
                                    'multiplicity': port['multiplicity'],
                                    'interfaces': [pi['name'] for pi in port[
                                        'interfaces']]
                                }) for port in
                            operation['ports']])
                    ports_list = operation['ports']
                    # Get operation requirements in tahiti
                    result = {
                        'N_INPUT': 0,
                        'N_OUTPUT': 0,
                        'PORT_NAMES': [],
                        'M_INPUT': 'None',
                        'M_OUTPUT': 'None'
                    }

                    # Correct form field types if the interface (Citron) does
                    # not send this information
                    for k, v in list(task.get('forms', {}).items()):
                        v['category'] = form_fields.get(k, 'EXECUTION')

                    for port in ports_list:
                        if port['type'] == 'INPUT':
                            result['M_INPUT'] = port['multiplicity']
                            if 'N_INPUT' in result:
                                result['N_INPUT'] += 1
                            else:
                                result['N_INPUT'] = 1
                        elif port['type'] == 'OUTPUT':
                            result['M_OUTPUT'] = port['multiplicity']
                            if 'N_OUTPUT' in result:
                                result['N_OUTPUT'] += 1
                            else:
                                result['N_OUTPUT'] = 1
                            if 'PORT_NAMES' in result:
                                result['PORT_NAMES'].append(
                                    (int(port['order']), port['name']))
                            else:
                                result['PORT_NAMES'] = [
                                    (int(port['order']), port['name'])]

                    port_names = [kv[1] for kv in sorted(
                        result['PORT_NAMES'], key=lambda _kv: _kv[0])]
                    task['port_names'] = port_names
                    self.graph.add_node(
                        task.get('id'),
                        name=task.get('name'),
                        forms=task.get('forms'),
                        operation=task.get('operation'),
                        in_degree_required=result['N_INPUT'],
                        in_degree_multiplicity_required=result['M_INPUT'],
                        out_degree_required=result['N_OUTPUT'],
                        out_degree_multiplicity_required=result['M_OUTPUT'],
                        port_names=port_names,
                        parents=[],
                        attr_dict=task)
                else:
                    msg = _("Task {task} uses an invalid or disabled "
                            "operation ({op})")
                    raise ValueError(
                        msg.format(task=task['id'], op=task['operation']['id']))
            else:
                self.disabled_tasks[task['id']] = task

        for flow in self.workflow['flows']:

            # Ignore disabled tasks
            if all([flow['source_id'] not in self.disabled_tasks,
                    flow['target_id'] not in self.disabled_tasks]):
                # Updates the source_port_name and target_port_name. They are
                # used in the transpiler part instead of the id of the port.
                source_port = list([p for p in
                                    task_map[flow['source_id']]['operation'][
                                        'ports'] if
                                    int(p['id']) == int(flow['source_port'])])

                target_port = list([p for p in
                                    task_map[flow['target_id']]['operation'][
                                        'ports'] if
                                    int(p['id']) == int(flow['target_port'])])

                if all([source_port, target_port]):
                    # Compatibility assertion, may be removed in future
                    # assert 'target_port_name' not in flow or \
                    #        flow['target_port_name'] == target_port[0]['slug']
                    # assert 'source_port_name' not in flow \
                    #      or flow['source_port_name'] == source_port[0]['slug']

                    flow['target_port_name'] = target_port[0]['slug']
                    flow['source_port_name'] = source_port[0]['slug']

                    self.graph.add_edge(flow['source_id'], flow['target_id'],
                                        attr_dict=flow)
                    self.graph.nodes[flow['target_id']]['parents'].append(
                        flow['source_id'])
                else:
                    self.log.warn(
                        _("Incorrect configuration for ports: %s, %s"),
                        source_port, target_port)
                    raise ValueError(_(
                        "Invalid or non-existing port: '{op}' {s} {t}").format(
                        op=task_map[flow['source_id']]['operation']['name'],
                        s=flow['source_port'], t=flow['target_port']))

        # for node in self.graph.nodes():
        #     self.graph.nodes[node]['in_degree'] = self.graph.in_degree(node)
        #
        #     self.graph.nodes[node]['out_degree'] = self.graph.out_degree(node)
        #     # self.graph.node[node]['parents'] = list(
        #     #        nx.edge_dfs(self.graph, node, orientation='reverse'))

        return self.graph

    # def builds_sorted_workflow_graph(self, tasks, flows):

    #     # Querying all operations from tahiti one time
    #     operations_tahiti = dict(
    #         [(op['id'], op) for op in self._get_operations()])
    #     for task in tasks:
    #         operation = operations_tahiti.get(task.get('operation')['id'])
    #         if operation is not None:
    #             ports_list = operations_tahiti[operation]['ports']
    #             # Get operation requirements in tahiti
    #             result = {
    #                 'N_INPUT': 0,
    #                 'N_OUTPUT': 0,
    #                 'M_INPUT': 'None',
    #                 'M_OUTPUT': 'None'
    #             }

    #             for port in ports_list:
    #                 if port['type'] == 'INPUT':
    #                     result['M_INPUT'] = port['multiplicity']
    #                     if 'N_INPUT' in result:
    #                         result['N_INPUT'] += 1
    #                     else:
    #                         result['N_INPUT'] = 1
    #                 elif port['type'] == 'OUTPUT':
    #                     result['M_OUTPUT'] = port['multiplicity']
    #                     if 'N_OUTPUT' in result:
    #                         result['N_OUTPUT'] += 1
    #                     else:
    #                         result['N_OUTPUT'] = 1
    #             # return result
    #             self.graph.add_node(
    #                 task.get('id'),
    #                 in_degree_required=result['N_INPUT'],
    #                 in_degree_multiplicity_required=result['M_INPUT'],
    #                 out_degree_required=result['N_OUTPUT'],
    #                 out_degree_multiplicity_required=result['M_OUTPUT'],
    #                 attr_dict=task)

    #     for flow in flows:
    #         self.graph.add_edge(flow['source_id'],
    #                             flow['target_id'],
    #                             attr_dict=flow)
    #         #parents = self.graph.node[flow['target_id']].get('parents', [])
    #         #parents.append(flow['source_id'])
    #         #self.graph.node[flow['target_id']]['parents'] = parents

    #     # updating in_degree and out_degree
    #     for node in self.graph.nodes():
    #         self.graph.node[node]['in_degree'] = self.graph. \
    #             in_degree(node)
    #         self.graph.node[node]['out_degree'] = self.graph. \
    #             out_degree(node)
    #         import pdb; pdb.set_trace()
    #         self.graph.node[node]['parents'] = list(
    #                 nx.edge_dfs(self.graph, node, orientation='reverse'))

    def plot_workflow_graph_image(self):
        """
             Show the image from workflow_graph
        """
        # Change layout according to necessity
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_color='#004a7b', node_size=2000,
                edge_color='#555555', width=1.5, edge_cmap=None,
                with_labels=True, style='dashed',
                label_pos=50.3, alpha=1, arrows=True, node_shape='s',
                font_size=8,
                font_color='#FFFFFF')

        # Must import pyplot here!
        import matplotlib.pyplot as plt
        plt.show()
        # If necessary save the image
        # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
        # bbox_inches=None, pad_inches=0.1)

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

        sorted_tasks_id = reversed(list(nx.topological_sort(self.graph)))

        return sorted_tasks_id

    def _is_there_null_source_id_tasks(self):
        for flow in self.workflow['flows']:
            if flow['source_id'] == "":
                return False
        return True

    def _is_there_null_target_id_tasks(self):
        for flow in self.workflow['flows']:
            if flow['target_id'] == "":
                return False
        return True

    def _get_operations(self, workflow_id):
        """ Returns operations available in Tahiti """
        tahiti_conf = self.config['juicer']['services']['tahiti']
        params = {
            'base_url': tahiti_conf['url'],
            'item_path': 'operations',
            'token': str(tahiti_conf['auth_token']),
            'item_id': '',
            'qs': 'workflow={}'.format(workflow_id)
        }

        # Querying tahiti operations to get number of inputs and outputs
        return tahiti_service.query_tahiti(params['base_url'],
                                           params['item_path'],
                                           params['token'],
                                           params['item_id'])

    def get_ports_from_operation_tasks(self, id_operation):
        tahiti_conf = self.config['juicer']['services']['tahiti']
        params = {
            'base_url': tahiti_conf['url'],
            'item_path': 'operations',
            'token': str(tahiti_conf['auth_token']),
            'item_id': id_operation
        }

        # Querying tahiti operations to get number of inputs and outputs
        operations_ports = tahiti_service.query_tahiti(params['base_url'],
                                                       params['item_path'],
                                                       params['token'],
                                                       params['item_id'])
        # Get operation requirements in tahiti
        result = {
            'N_INPUT': 0,
            'N_OUTPUT': 0,
            'M_INPUT': 'None',
            'M_OUTPUT': 'None'
        }

        for port in operations_ports['ports']:
            if port['type'] == 'INPUT':
                result['M_INPUT'] = port['multiplicity']
                if 'N_INPUT' in result:
                    result['N_INPUT'] += 1
                else:
                    result['N_INPUT'] = 1
            elif port['type'] == 'OUTPUT':
                result['M_OUTPUT'] = port['multiplicity']
                if 'N_OUTPUT' in result:
                    result['N_OUTPUT'] += 1
                else:
                    result['N_OUTPUT'] = 1
        return result

    def workflow_execution_parcial(self):

        topological_sort = self.get_topological_sorted_tasks()

        for node_obj in topological_sort:
            print(nx.ancestors(self.graph, node_obj),
                  self.graph.predecessors(node_obj),
                  node_obj,
                  self.graph.nodes[node_obj]['in_degree_required'],
                  self.graph.nodes[node_obj]['in_degree'],
                  self.graph.nodes[node_obj]['out_degree_required'],
                  self.graph.nodes[node_obj]['out_degree']
                  )
        return True

        # # only to debug
        # def check_outdegree_edges(self, atr):
        #
        #     if self.graph.has_node(atr):
        #         return (self.graph.node[atr]['in_degree'],
        #                 self.graph.node[atr]['out_degree'],
        #                 self.graph.in_degree(atr),
        #                 self.graph.out_degree(atr),
        #                 self.graph.node[atr]['in_degree_required'],
        #                 self.graph.node[atr]['out_degree_required']
        #                 )
        #     if self.query_operations:
        #         return self.query_operations()
    def handle_variables(self, custom_vars=None):
        """
        Handles variable substitution
        """
        now = datetime.datetime.now()
        date_at_min = datetime.datetime.combine(datetime.datetime.now(), datetime.time.min)
        date_at_max = datetime.datetime.combine(datetime.datetime.now(), datetime.time.max)

        all_vars = {
            'date': now.strftime('%Y-%m-%d'),
            'now': now.strftime('%Y-%m-%d %H:%M:%S'),
            'date_at_min': date_at_min.strftime('%Y-%m-%d %H:%M:%S'),
            'date_at_max': date_at_max.strftime('%Y-%m-%d %H:%M:%S'),
            'user_login': self.workflow['user']['login'],
            'user_name': self.workflow['user']['name'],
            'user_email': self.workflow['user']['login'], # FIXME
            'user_id': str(self.workflow['user']['id']),
            'workflow_name': self.workflow['name'],
            'workflow_id': self.workflow['id'],
        }
        if custom_vars:
            all_vars.update(custom_vars)

        for variable in self.workflow.get('variables'):
            all_vars[variable['name']] = variable.get('value', variable.get('default_value'))

        variable_re = re.compile(r'\$\{[_A-Za-z][_A-Za-z0-9]*\}')
        for task in self.workflow['tasks']:
            if 'forms' in task and task['enabled']:
                for k, v in list(task.get('forms').items()):
                    value = v.get('value')
                    if task['operation']['slug'] == 'user-filter' and \
                            k == 'filters': # FIXME: Needs to be dynamic
                        for filter_value in value:
                            value1 = str(filter_value.get('value', filter_value.get('default_value', '')))
                            filter_value['value'] = value1
                            for found in variable_re.findall(value1):
                                var_name = found[2:-1]
                                if var_name in all_vars:
                                    new_value = str(all_vars[var_name])
                                    filter_value['value'] = filter_value.get(
                                            'value').replace(found, new_value)
                                else:
                                    raise ValueError(
                                         _('Variable "{}" is used in task "{}", but it is undefined.'.format(
                                          var_name, task['name'])))

                    if isinstance(value, (str,)):
                        for found in variable_re.findall(value):
                            var_name = found[2:-1]
                            if var_name in all_vars:
                                new_value = str(all_vars[var_name])
                                v['value'] = v.get('value').replace(found, new_value)
                            else:
                                raise ValueError(
                                     _('Variable "{}" is used in task "{}", but it is undefined.'.format(
                                     var_name, task['name'])))
            
