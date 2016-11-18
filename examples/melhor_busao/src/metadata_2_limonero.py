
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName('## databases_melhor_busao ##') \
    .getOrCreate()

# READ_CSV
databases_melhor_busao_df_0 = spark.read.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ibge.csv',
            header=True, sep=';' ,inferSchema=True)
print "READ_CSV" 
databases_melhor_busao_df_0.show()

# SAVE
databases_melhor_busao_df_0.write.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ibge.csv', header=True, mode='overwrite')
from metadata import MetadataPost

types_names = dict()
types_names['IntegerType'] = "INTEGER"
types_names['StringType'] = "TEXT"
types_names['LongType'] = "LONG"
types_names['DoubleType'] = "DOUBLE"
types_names['TimestampType'] = "DATETIME"


schema = []
for att in databases_melhor_busao_df_0.schema:
    data = dict()
    data['name'] = att.name
    data['dataType'] = types_names[str(att.dataType)]
    data['nullable'] = att.nullable
    data['metadata'] = att.metadata
    schema.append(data)

parameters = dict()
parameters['name'] = "ibge"
parameters['format'] = "CSV"
parameters['storage_id'] = 1
parameters['provenience'] = str("{'user': {'login': 'fernando', 'id': 432, 'name': 'Fernando Carvalho'}, 'workflow': {'framework': 'Spark', 'tasks': [{'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '001', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ticketing.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 1}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '002', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'ticketing'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ticketing.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['ticketing', 'curitiba', 'bus']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 1}]}, {'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '003', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/gps.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 2}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '004', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'gps'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/gps.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['gps', 'curitiba', 'bus']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 2}]}, {'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '005', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ibge.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 3}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '006', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'ibge'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ibge.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['ibge', 'economics', 'social']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 3}]}], 'id': 223, 'name': 'databases_melhor_busao'}}")
parameters['description'] = "databases_melhor_busao"
parameters['user_id'] = "432"
parameters['user_login'] = "fernando"
parameters['user_name'] = "Fernando Carvalho"
parameters['workflow_id'] = "223"
parameters['url'] = "hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ibge.csv"

instance = MetadataPost('123456', schema, parameters)


# READ_CSV
databases_melhor_busao_df_1 = spark.read.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ticketing.csv',
            header=True, sep=';' ,inferSchema=True)
print "READ_CSV" 
databases_melhor_busao_df_1.show()

# SAVE
databases_melhor_busao_df_1.write.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ticketing.csv', header=True, mode='overwrite')
from metadata import MetadataPost

types_names = dict()
types_names['IntegerType'] = "INTEGER"
types_names['StringType'] = "TEXT"
types_names['LongType'] = "LONG"
types_names['DoubleType'] = "DOUBLE"
types_names['TimestampType'] = "DATETIME"


schema = []
for att in databases_melhor_busao_df_1.schema:
    data = dict()
    data['name'] = att.name
    data['dataType'] = types_names[str(att.dataType)]
    data['nullable'] = att.nullable
    data['metadata'] = att.metadata
    schema.append(data)

parameters = dict()
parameters['name'] = "ticketing"
parameters['format'] = "CSV"
parameters['storage_id'] = 1
parameters['provenience'] = str("{'user': {'login': 'fernando', 'id': 432, 'name': 'Fernando Carvalho'}, 'workflow': {'framework': 'Spark', 'tasks': [{'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '001', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ticketing.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 1}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '002', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'ticketing'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ticketing.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['ticketing', 'curitiba', 'bus']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 1}]}, {'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '003', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/gps.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 2}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '004', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'gps'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/gps.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['gps', 'curitiba', 'bus']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 2}]}, {'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '005', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ibge.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 3}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '006', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'ibge'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ibge.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['ibge', 'economics', 'social']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 3}]}], 'id': 223, 'name': 'databases_melhor_busao'}}")
parameters['description'] = "databases_melhor_busao"
parameters['user_id'] = "432"
parameters['user_login'] = "fernando"
parameters['user_name'] = "Fernando Carvalho"
parameters['workflow_id'] = "223"
parameters['url'] = "hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ticketing.csv"

instance = MetadataPost('123456', schema, parameters)


# READ_CSV
databases_melhor_busao_df_2 = spark.read.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/gps.csv',
            header=True, sep=';' ,inferSchema=True)
print "READ_CSV" 
databases_melhor_busao_df_2.show()

# SAVE
databases_melhor_busao_df_2.write.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/gps.csv', header=True, mode='overwrite')
from metadata import MetadataPost

types_names = dict()
types_names['IntegerType'] = "INTEGER"
types_names['StringType'] = "TEXT"
types_names['LongType'] = "LONG"
types_names['DoubleType'] = "DOUBLE"
types_names['TimestampType'] = "DATETIME"


schema = []
for att in databases_melhor_busao_df_2.schema:
    data = dict()
    data['name'] = att.name
    data['dataType'] = types_names[str(att.dataType)]
    data['nullable'] = att.nullable
    data['metadata'] = att.metadata
    schema.append(data)

parameters = dict()
parameters['name'] = "gps"
parameters['format'] = "CSV"
parameters['storage_id'] = 1
parameters['provenience'] = str("{'user': {'login': 'fernando', 'id': 432, 'name': 'Fernando Carvalho'}, 'workflow': {'framework': 'Spark', 'tasks': [{'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '001', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ticketing.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 1}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '002', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'ticketing'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ticketing.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['ticketing', 'curitiba', 'bus']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 1}]}, {'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '003', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/gps.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 2}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '004', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'gps'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/gps.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['gps', 'curitiba', 'bus']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 2}]}, {'operation': {'name': 'READ_CSV', 'id': 434}, 'log_level': 'INFO', 'id': '005', 'parameters': [{'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/tests/ibge.csv'}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'separator', 'value': ';'}], 'ports': [{'interface': 'dataframe', 'direction': 'out', 'id': 3}]}, {'operation': {'name': 'SAVE', 'id': 2001}, 'log_level': 'INFO', 'id': '006', 'parameters': [{'category': 'EXECUTION', 'name': 'name', 'value': 'ibge'}, {'category': 'EXECUTION', 'name': 'url', 'value': 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ibge.csv'}, {'category': 'EXECUTION', 'name': 'format', 'value': 'CSV'}, {'category': 'EXECUTION', 'name': 'storage_id', 'value': 1}, {'category': 'EXECUTION', 'name': 'header', 'value': 'True'}, {'category': 'EXECUTION', 'name': 'mode', 'value': 'overwrite'}, {'category': 'EXECUTION', 'name': 'tags', 'value': '['ibge', 'economics', 'social']'}], 'ports': [{'interface': 'dataframe', 'direction': 'in', 'id': 3}]}], 'id': 223, 'name': 'databases_melhor_busao'}}")
parameters['description'] = "databases_melhor_busao"
parameters['user_id'] = "432"
parameters['user_login'] = "fernando"
parameters['user_name'] = "Fernando Carvalho"
parameters['workflow_id'] = "223"
parameters['url'] = "hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/gps.csv"

instance = MetadataPost('123456', schema, parameters)

