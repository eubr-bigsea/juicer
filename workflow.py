import json
import networkx as nx

import matplotlib as matpl
matpl.use('Agg')
import matplotlib.pyplot as plt
import Tkinter


class Workflow:

    def __init__(self, infile):
        # Infile with the workflow json
        self.infile = infile
        # Workflow dictionary
        self.workflow = {}
        # Topological sorted tasks according to their dependencies
        self.sorted_tasks = []


    def read_json(self):
        ''' Opens the Json file and build the workflow dict '''
        with open(self.infile) as json_infile:
            self.workflow = json.load(json_infile)


    def verify_workflow(self):
        ''' Verifies if the workflow is valid '''
        # IMPLEMENT!
        pass


    def sort_tasks(self):
        ''' Create the tasks Graph and perform topological sorting '''
        # First, map the tasks IDs to their original position
        tasks_position = {}
        count_position = 0
        for i in range(0, len(self.workflow['tasks'])):
            tasks_position[self.workflow['tasks'][i]['id']] = count_position
            count_position += 1
        # Then, performs topological sorting
        workflow_graph = self.builds_workflow_graph(self.workflow['tasks'])
        sorted_tasks_id = nx.topological_sort(workflow_graph, reverse=False)
        # Finally, create a new array of tasks in the topogical order
        for task_id in sorted_tasks_id:
            self.sorted_tasks.append(self.workflow['tasks'][tasks_position[task_id]])


    def plot_workflow(self, filename):
        ''' Plot the workflow graph '''
        workflow_graph = self.builds_workflow_graph(self.sorted_tasks)
        pos = nx.spectral_layout(workflow_graph)
        nx.draw(workflow_graph, pos, node_color='#004a7b', node_size=2000,
                edge_color='#555555', width=1.5, edge_cmap=None,with_labels=True,
                label_pos=50.3, alpha=1, arrows=True, node_shape='s', font_size=8,
                font_color='#FFFFFF')
        plt.savefig(filename, dpi=300, orientation='landscape', format=None,
                    bbox_inches=None, pad_inches=0.1)


    def builds_workflow_graph(self, tasks):
        ''' Builds a graph with the tasks '''
        workflow_graph = nx.DiGraph()
        edges = {}
        edge_labels = {}
        for task in tasks:
            workflow_graph.add_node(task['id'])
            for port in task['ports']:
                if not edges.has_key(port['id']):
                    edges[port['id']] = {}
                edges[port['id']][port['type']] = task['id']
        for port in edges:
            if (edges[port].has_key('in') and edges[port].has_key('out')):
                workflow_graph.add_edge(edges[port]['out'], edges[port]['in'])
                #edge_labels[(edges[port]['out'], edges[port]['in'])] = port
        return workflow_graph