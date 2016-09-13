from pyspark.sql import SparkSession
import sys
import argparse
import json
import addresser

args = {}
workflow = {}
response = {}
response['tasks'] = []
ports = {}

def read_parameters():
    ''' READS THE PARAMETERS AND LOADS TO THE ARGS VAR'''
    global args
    parser = argparse.ArgumentParser(description='Lemonade module that receive the workflow Json and generate the Spark code.')
    parser.add_argument('-j', '--json', help='Json file describing the Lemonade workflow', required=True)
#    parser.add_argument('-i', '--infile', help='Infile location', required=True)
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

    print "BEFORE CREATING SESSION"
    spark = SparkSession\
        .builder\
        .appName("## Lemonade_workflow_consumer ##")\
        .getOrCreate()
    print "AFTER CREATING SESSION"

    for task in workflow['tasks']:
        print "Running task",task['operation_name']
        addresser.addresser(spark, task, ports)


