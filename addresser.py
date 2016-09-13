import operation

def addresser(spark, task, ports):

    if (task['operation_name'] == "DATA_READER"):
        # CORRECT: query file management API to load database
        leitor = operation.DataReader(task['parameters'][0]['infile'])
        df = leitor.read_csv(task['parameters'][0]['has_header'], task['parameters'][0]['sep'], spark)
        df.show()

    elif (task['operation_name'] == 'RANDOM_SPLIT'):
        spliter = operation.RandomSplit(task['parameters'][0]['weights'], task['parameters'][0]['seed'])
        split_df = spliter.split(df)

#    elif (task['id'] == 'DISTINCT_UNION'):
#        distinct_union(task, response)
