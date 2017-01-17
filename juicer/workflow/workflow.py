import networkx as nx
import matplotlib.pyplot as plt

class Workflow:
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

    def __init__(self, workflow_data):

        self.workflow_graph = nx.MultiDiGraph()

        # self.graph = nx.MultiDiGraph()

        # Workflow dictionary
        self.workflow_data = workflow_data

        self.workflow_graph = self.builds_initial_workflow_graph()

        # Topological sorted tasks according to their dependencies
        self.sorted_tasks = []

        self.sort_tasks()


    def builds_initial_workflow_graph(self):
        """ Builds a graph with the tasks """

        for task in self.workflow_data['tasks']:
            self.workflow_graph.add_node(task.get('id'), attr_dict=task)

        for flow in self.workflow_data['flows']:
            self.workflow_graph.add_edge(flow['source_id'], flow['target_id'],
                                attr_dict=flow)

        return self.workflow_graph

    def builds_sorted_workflow_graph(self, tasks, flows):

        workflow_graph = nx.MultiDiGraph()

        for task in tasks:
            workflow_graph.add_node(task.get('id'), attr_dict=task)

        for flow in flows:
            workflow_graph.add_edge(flow['source_id'], flow['target_id'],
                                         attr_dict=flow)

        return workflow_graph


    def plot_workflow_graph_image(self):
        # Change layout according to necessity
        pos = nx.spring_layout(self.workflow_graph)
        nx.draw(self.workflow_graph, pos, node_color='#004a7b', node_size=2000,
                edge_color='#555555', width=1.5, edge_cmap=None,
                with_labels=True, style='dashed',
                label_pos=50.3, alpha=1, arrows=True, node_shape='s',
                font_size=8,
                font_color='#FFFFFF')
        plt.show()
        # If necessary save the image
        # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
        # bbox_inches=None, pad_inches=0.1)

    def sort_tasks(self):
        """ Create the tasks Graph and perform topological sorting """
        # First, map the tasks IDs to their original position
        tasks_position = {}

        for count_position, task in enumerate(self.workflow_data['tasks']):
            tasks_position[task['id']] = count_position


        # Then, performs topological sorting
        # workflow_graph_aux = self.builds_workflow_graph()
        # workflow_graph_aux = self.builds_sorted_workflow_graph(
        #     self.workflow_data['tasks'], self.workflow_data['flows'])

        sorted_tasks_id = nx.topological_sort(self.workflow_graph, reverse=False)

        # self.sorted_tasks = tasks_position
        # self.sorted_tasks = sorted_tasks_id


        # Finally, create a new array of tasks in the topogical order
        for task_id in sorted_tasks_id:
            self.sorted_tasks.append(self.workflow_data['tasks']
                                     [tasks_position[task_id]])

    # Need to be implemented
    def topological_sorted_tasks(self):
        return 1

    def check_null_source_id_tasks(self):
        for flow in self.workflow_data['flows']:
            if flow['source_id'] is None:
                print("Existem tarefas nulas")
                return 0
            else:
                pass

        return 1

    def check_null_target_id_tasks(self):
        return 0

    # def verify_workflow(self):
    #     """
    #     Verifies if the workflow is valid.
    #     Validations to be implemented:
    #     - Supported platform
    #     - Workflow without input
    #     - Supported operations in platform
    #     - Consistency between tasks and flows
    #     - Port consistency
    #     - Task parameters
    #     - Referenced attributes names existing in input dataframes
    #     """
    #     pass

        # def sort_tasks(self):
        #     """ Create the tasks Graph and perform topological sorting """
        #     # First, map the tasks IDs to their original position
        #     tasks_position = {}
        #
        #     for count_position, task in enumerate(self.workflow['tasks']):
        #         tasks_position[task['id']] = count_position
        #
        #     # Then, performs topological sorting
        #     workflow_graph = self.builds_workflow_graph(
        #         self.workflow['tasks'], self.workflow['flows'])
        #     sorted_tasks_id = nx.topological_sort(workflow_graph, reverse=False)
        #     # Finally, create a new array of tasks in the topogical order
        #     for task_id in sorted_tasks_id:
        #         self.sorted_tasks.append(
        #             self.workflow['tasks'][tasks_position[task_id]])

        # def plot_workflow(self, filename):
        #    """ Plot the workflow graph """
        # workflow_graph = self.builds_workflow_graph(self.sorted_tasks,
        #                                                self.workflow['flows'])
        # pos = nx.spring_layout(workflow_graph)
        # nx.draw(workflow_graph, pos, node_color='#004a7b', node_size=2000,
        #         edge_color='#555555', width=1.5, edge_cmap=None,
        #         with_labels=True,
        #         label_pos=50.3, alpha=1, arrows=True, node_shape='s',
        #         font_size=8,
        #         font_color='#FFFFFF')
        # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
        #             bbox_inches=None, pad_inches=0.1)

        # def builds_workflow_graph(self, tasks, flows):
        #     """ Builds a graph with the tasks """
        #     workflow_graph = nx.DiGraph()
        #
        #     for task in tasks:
        #         workflow_graph.add_node(task['id'])
        #
        #     for flow in flows:
        #         workflow_graph.add_edge(flow['source_id'], flow['target_id'])
        #     return workflow_graph
