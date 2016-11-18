
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName('## melhor_busao ##') \
    .getOrCreate()

# DATA_READER
melhor_busao_df_0 = spark.read.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ticketing.csv',
            header=True, sep=',' ,inferSchema=True)
print "DATA_READER Read_Ticketing_data" 
melhor_busao_df_0.show()

# DATA_READER
melhor_busao_df_1 = spark.read.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/ibge.csv',
            header=True, sep=',' ,inferSchema=True)
print "DATA_READER Read_IBGE_data" 
melhor_busao_df_1.show()

# DATA_READER
melhor_busao_df_2 = spark.read.csv('hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/melhor_busao/gps.csv',
            header=True, sep=',' ,inferSchema=True)
print "DATA_READER Read_GPS_data" 
melhor_busao_df_2.show()

# FILTER
melhor_busao_df_3 = melhor_busao_df_2.filter('COD_LINHA == 801')
print "FILTER Filter_CODLINHA_GPS" 
melhor_busao_df_3.show()

# DATETIME_TO_BINS

from bins import *
melhor_busao_df_4 = datetime_to_bins(melhor_busao_df_3, 5, 'DTHR', 'BINS_5_MIN')

print "DATETIME_TO_BINS Bins_GPS" 
melhor_busao_df_4.show()

# AGGREGATION
melhor_busao_df_5 = melhor_busao_df_4.groupBy(['COD_LINHA', 'VEIC', 'BINS_5_MIN']).agg(first('LAT').alias('LAT'), first('LON').alias('LON'))
print "AGGREGATION Aggregation_GPS" 
melhor_busao_df_5.show()

# FILTER
melhor_busao_df_6 = melhor_busao_df_0.filter('CODLINHA == 801')
print "FILTER Filter_CODLINHA_Ticketing" 
melhor_busao_df_6.show()

# DATETIME_TO_BINS

from bins import *
melhor_busao_df_7 = datetime_to_bins(melhor_busao_df_6, 5, 'DATAUTILIZACAO', 'BINS_5_MIN')

print "DATETIME_TO_BINS Bins_Ticketing" 
melhor_busao_df_7.show()

# SELECT
melhor_busao_df_8 = melhor_busao_df_7.select(['CODLINHA', 'CODVEICULO', 'BINS_5_MIN', 'NUMEROCARTAO', 'DATAUTILIZACAO'])
print "SELECT SELECT_Ticketing" 
melhor_busao_df_8.show()

# SORT
melhor_busao_df_9 = melhor_busao_df_8.orderBy(["CODLINHA", "CODVEICULO", "BINS_5_MIN", "DATAUTILIZACAO"], ascending=["True", "True", "True", "True"])
print "SORT SORT_Ticketing" 
melhor_busao_df_9.show()

# JOIN

cond_melhor_busao_df_10 = [melhor_busao_df_9.CODLINHA == melhor_busao_df_5.COD_LINHA, melhor_busao_df_9.CODVEICULO == melhor_busao_df_5.VEIC, melhor_busao_df_9.BINS_5_MIN == melhor_busao_df_5.BINS_5_MIN]
melhor_busao_df_10 = melhor_busao_df_9.join(melhor_busao_df_5, on=cond_melhor_busao_df_10, how='inner').drop(melhor_busao_df_5.COD_LINHA).drop(melhor_busao_df_5.VEIC).drop(melhor_busao_df_5.BINS_5_MIN)
print "JOIN Join_Ticketing_GPS" 
melhor_busao_df_10.show()

# SORT
melhor_busao_df_11 = melhor_busao_df_10.orderBy(["NUMEROCARTAO", "DATAUTILIZACAO"], ascending=["True", "True"])
print "SORT SORT_Ticketing_GPS" 
melhor_busao_df_11.show()

# AGGREGATION
melhor_busao_df_12 = melhor_busao_df_11.groupBy(['NUMEROCARTAO']).agg(count('NUMEROCARTAO').alias('COUNT'))
print "AGGREGATION Aggregation_count_Ticket_GPS" 
melhor_busao_df_12.show()
