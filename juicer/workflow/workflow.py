import networkx as nx


class Workflow:
    def __init__(self, workflow_data):
        self.graph = nx.DiGraph()
        self.workflow = workflow_data
        for task in workflow_data['tasks']:
            self.graph.add_node(task.get('id'), attr_dict=task)

        for flow in workflow_data['flows']:
            self.graph.add_edge(flow['source_id'], flow['target_id'],
                                attr_dict=flow)

        # Workflow dictionary
        # self.workflow = workflow_data
        # Topological sorted tasks according to their dependencies
        # self.sorted_tasks = []

    # def set_workflow(self, workflow):
    #     self.workflow = workflow

    def verify_workflow(self):
        """
        Verifies if the workflow is valid.
        Validations to be implemented:
        - Supported platform
        - Workflow without input
        - Supported operations in platform
        - Consistency between tasks and flows
        - Port consistency
        - Task parameters
        - Referenced attributes names existing in input dataframes
        """
        pass

    def sort_tasks(self):
        """ Create the tasks Graph and perform topological sorting """
        # First, map the tasks IDs to their original position
        tasks_position = {}

        for count_position, task in enumerate(self.workflow['tasks']):
            tasks_position[task['id']] = count_position

        # Then, performs topological sorting
        workflow_graph = self.builds_workflow_graph(
            self.workflow['tasks'], self.workflow['flows'])
        sorted_tasks_id = nx.topological_sort(workflow_graph, reverse=False)
        # Finally, create a new array of tasks in the topogical order
        for task_id in sorted_tasks_id:
            self.sorted_tasks.append(
                self.workflow['tasks'][tasks_position[task_id]])

    def plot_workflow(self, filename):
        """ Plot the workflow graph """
        workflow_graph = self.builds_workflow_graph(self.sorted_tasks,
                                                    self.workflow['flows'])
        # pos = nx.spring_layout(workflow_graph)
        # nx.draw(workflow_graph, pos, node_color='#004a7b', node_size=2000,
        #         edge_color='#555555', width=1.5, edge_cmap=None,
        #         with_labels=True,
        #         label_pos=50.3, alpha=1, arrows=True, node_shape='s',
        #         font_size=8,
        #         font_color='#FFFFFF')
        # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
        #             bbox_inches=None, pad_inches=0.1)

    def builds_workflow_graph(self, tasks, flows):
        """ Builds a graph with the tasks """
        workflow_graph = nx.DiGraph()

        for task in tasks:
            workflow_graph.add_node(task['id'])

        for flow in flows:
            workflow_graph.add_edge(flow['source_id'], flow['target_id'])
        return workflow_graph
