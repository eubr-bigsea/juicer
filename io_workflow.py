import sys
import argparse
import json
import Tkinter
import networkx as nx
import matplotlib as matpl
matpl.use('Agg')
import matplotlib.pyplot as plt
from juicer.spark import operation


classes = {}
classes['DATA_READER'] = operation.DataReader
classes['RANDOM_SPLIT'] = operation.RandomSplit
classes['UNION'] = operation.Union
classes['SORT'] = operation.Sort
classes['SAVE'] = operation.Save


def read_parameters():
    ''' Read the parameters and return the dict'''
    global args
    parser = argparse.ArgumentParser(description='Lemonade module that receive \
        the workflow Json and generate the Spark code.')
    parser.add_argument('-j', '--json', help='Json file describing the Lemonade \
        workflow', required=True)
    parser.add_argument('-o', '--outfile', help = 'Outfile name to receive the Spark \
        code', required=True)
    return vars(parser.parse_args())


def read_json(json_file_name):
    ''' Opens the Json and returns the dict '''
    with open(json_file_name) as json_infile:
        workflow = json.load(json_infile)
    return workflow


def print_session(output, workflow_id):
    ''' Print the PySpark header and session init  '''
    output.write("from pyspark.sql import SparkSession \n\n")
    output.write("spark = SparkSession\\\n")
    output.write("    .builder\\\n")
    output.write("    .appName('## "+workflow_id+" ##')\\\n")
    output.write("    .getOrCreate()\n\n")


def topological_sorting(workflow):
    ''' Create the tasks Graph and perform topological sorting '''
    workflow_graph = nx.DiGraph()
    edges = {}
    edge_labels = {}
    for task in workflow['tasks']:
        workflow_graph.add_node(task['id'])
        for port in task['ports']:
            if not edges.has_key(port['id']):
                edges[port['id']] = {}
            edges[port['id']][port['type']] = task['id']
    for port in edges:
        if (edges[port].has_key('in') and edges[port].has_key('out')):
            workflow_graph.add_edge(edges[port]['out'], edges[port]['in'])
            edge_labels[(edges[port]['out'], edges[port]['in'])] = port

    return nx.topological_sort(workflow_graph, reverse=False)


def map_task_position(workflow):
    ''' Map the position of each task in the workflow '''
    count_position = 0
    map_position = {}
    for i in range(0,len(workflow['tasks'])):
        map_position[workflow['tasks'][i]['id']] = count_position
        count_position += 1
    return map_position


def sort_tasks(workflow, map_position):
    ''' Create a new array of tasks in the topogical order '''
    task_sorted = []
    for task_id in topological_sorting(workflow):
        task_sorted.append(workflow['tasks'][map_position[task_id]])
    return task_sorted


def map_port_dataframe(task, ports_df, df_output, df_input, count_df):
    ''' Map each port of the task with a dataframe '''
    for port in task['ports']:
        # IF PORT IS OUT, CREATE A DATAFRAME NAME AND INCREMENT THE COUNTER
        if port['type'] == 'out':
            ports_df[port['id']] = workflow['name'] + '_df_' + str(count_df)
            df_output.append(ports_df[port['id']])
            count_df += 1
        # IF PORT IS IN, RETRIEVE THE DATAFRAME NAME
        else:
            df_input.append(ports_df[port['id']])


def juicer_excution(workflow, task_sorted, outfile_name):
    ''' Executes the tasks in Lemonade's workflow '''
    output = open(outfile_name, 'w')
    print_session(output, workflow['name'])
    # For each port connection, there is a data frame
    ports_df = {}
    count_df = 0
    for task in task_sorted:
        output.write("\n# " + task['operation_name'] + "\n")
        # BEFORE OPERATION, CREATE A RECORD FOR EACH OUT PORT AND
        # CREATE AN ARRAY WITH THE IN AND OUT DATAFRAMES RELATED TO THE OPERATION PORTS
        df_input = []
        df_output = []
        map_port_dataframe(task, ports_df, df_output, df_input, count_df)
        # CREATE THE OBJECT
        class_name = classes[task['operation_name']]
        instance = class_name(task['parameters'][0], df_input, df_output)
        # GENERATE THE SPARK CODE
        output.write(instance.generate_code() + "\n")
        for out in df_output:
            output.write("print \"" + task['operation_name'] + "\" \n")
            output.write(out + ".show()\n")


if __name__ == '__main__':
    args = read_parameters()
    workflow = read_json(args['json'])
    #verify_json()
    #json_validation()
    #user_authentication()
    #create_log()
    task_position = map_task_position(workflow)
    task_sorted = sort_tasks(workflow, task_position)
    juicer_excution(workflow, task_sorted, args['outfile'])
