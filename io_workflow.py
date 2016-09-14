import sys
import argparse
import json
import operation

classes = {}
classes['DATA_READER'] = operation.DataReader
classes['RANDOM_SPLIT'] = operation.RandomSplit
classes['DISTINCT_UNION'] = operation.Union

args = {}
workflow = {}
response = {}
response['tasks'] = []
# For each port connection, there is a data frame
ports_df = {}
count_df = 0


def read_parameters():
    ''' READS THE PARAMETERS AND LOADS TO THE ARGS VAR'''
    global args
    parser = argparse.ArgumentParser(description='Lemonade module that receive the workflow Json and generate the Spark code.')
    parser.add_argument('-j', '--json', help='Json file describing the Lemonade workflow', required=True)
    parser.add_argument('-o', '--outfile', help = 'Outfile name to receive the Spark code', required=True)
    args = vars(parser.parse_args())


def read_json():
    ''' OPENS THE JSON AND LOADS IT TO THE JSON VAR'''
    global workflow
    with open(args['json']) as json_infile:
        workflow = json.load(json_infile)

def print_session(output):
    output.write("from pyspark.sql import SparkSession \n\n")
    output.write("spark = SparkSession.builder.appName('## Lemonade_workflow_consumer ##').getOrCreate()")
    output.write("\n\n")
    #output.write()

if __name__ == '__main__':
    read_parameters()
    read_json()
    #user_authentication()
    #create_log()
    output = open(args['outfile'], 'w')
    print_session(output)

    for task in workflow['tasks']:
        output.write("\n#" + task['operation_name'] + "\n")

        # BEFORE OPERATION, CREATE A RECORD FOR EACH OUT PORT AND
        # CREATE AN ARRAY WITH THE IN AND OUT DATAFRAMES RELATED TO THE OPERATION PORTS
        df_input = []
        df_output = []
        for port in task['ports']:
            # IF PORT IS OUT, CREATE A DATAFRAME NAME AND INCREMENT THE COUNTER
            if port['type'] == 'out':
                ports_df[port['id']] = workflow['name'] + '_df_' + str(count_df)
                df_output.append(ports_df[port['id']])
                count_df += 1
            # IF PORT IS IN, RETRIEVE THE DATAFRAME NAME
            else:
                df_input.append(ports_df[port['id']])

        # CREATE THE OBJECT
        class_name = classes[task['operation_name']]
        instance = class_name(task['parameters'][0], df_input, df_output)

        # GENERATE THE SPARK CODE
        output.write(instance.generate_code() + "\n")
        
        for out in df_output:
            output.write("print \"" + task['operation_name'] + "\" \n")
            output.write(out + ".show()\n") 
