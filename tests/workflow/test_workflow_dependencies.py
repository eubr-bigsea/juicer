# -*- coding: utf-8 -*-

import pytest
import networkx as nx
import json
import matplotlib.pyplot as plt

from juicer.workflow.workflow import Workflow

def debug_instance(instance_wf):
    print
    print '*' * 20
    print instance_wf.workflow_graph.nodes()
    print '*' * 20
    print instance_wf.workflow_graph.edges()
    print '*' * 20
    print instance_wf.workflow_graph.is_multigraph()
    print '*' * 20
    print instance_wf.workflow_graph.number_of_edges()
    print '*' * 20
    print instance_wf.sorted_tasks
    print '*' * 20
    test = instance_wf.get_reversed_graph()
    print test.edges()
    print '*' * 20
    print nx.topological_sort(test, reverse=False)
    print '*' * 40
    print instance_wf.get_ports_from_operation_tasks()
    print '*' * 40

    ## Show image
    # pos = nx.spring_layout(instance_wf.workflow_graph)
    # pos = nx.fruchterman_reingold_layout(instance_wf.graph)
    # nx.draw(instance_wf.workflow_graph, pos, node_color='#004a7b', node_size=2000,
    #         edge_color='#555555', width=1.5, edge_cmap=None,
    #         with_labels=True, style='dashed',
    #         label_pos=50.3, alpha=1, arrows=True, node_shape='s',
    #         font_size=8,
    #         font_color='#FFFFFF')
    # plt.show()
    # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
                 # bbox_inches=None, pad_inches=0.1)

def test_workflow_sequence_success():

    # workflow_completo
    workflow_test = json.load(open("./tests/workflow/workflow_correct_changedid.txt"),
                              encoding='utf-8')

    # workflow_test = json.dumps(json.load(
        # open("./tests/workflow/workflow_correct_changedid.txt"), encoding='utf-8'))

    instance_wf = Workflow(workflow_test)

    # sorted_tasks_id = nx.topological_sort(instance_wf, reverse=False)
    # print sorted_tasks_id

    print debug_instance(instance_wf)
    assert instance_wf, debug_instance(instance_wf)

def test_workflow_sequence_missing_targetid_value_failure():

    # workflow with missing target_id
    workflow_test = json.load(open("./tests/workflow/workflow_missing_targetid.txt"))
    with pytest.raises(AttributeError):
        Workflow(workflow_test)

def test_workflow_sequence_missing_sourceid_value_failure():

    # workflow with missing target_id
    workflow_test = json.load(open("./tests/workflow/workflow_missing_sourceid.txt"))
    with pytest.raises(AttributeError):
        Workflow(workflow_test)