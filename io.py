import sys
import argparse
import json
from addresser import addresser

args = {}
workflow = {}
response = {}
ports = {}

def read_parameters():
    ''' READS THE PARAMETERS AND LOADS TO THE ARGS VAR'''
    global args
    parser = argparse.ArgumentParser(description='Lemonade module that receive the workflow Json and generate the Spark code.')
    parser.add_argument('-j', '--json', help='Json file describing the Lemonade workflow', required=True)
    parser.add_argument('-i', '--infile', help='Infile location', required=True)
    args = vars(parser.parse_args())


def read_json():
    ''' OPENS THE JSON AND LOADS IT TO THE JSON VAR'''
    global workflow
    with open(args['json']) as json_infile:
        workflow = json.load(json_infile)

# Se a porta eh do tipo out, preenche informacao
# Se a porta eh do tipo in, carrega informacao

if __name__ == '__main__':
    read_parameters()
    read_json()
    #user_authentication()
    #create_log()
    for task in workflow['tasks']:
        #task['input_location'] = get_database_location(workflow['user'],task['input_id'])
        #task['input_location'] = new_location(workflow['user'],task['output_name_id'])
        addresser(task, ports)
