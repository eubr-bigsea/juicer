#!/usr/bin/env python
"""
Auto-generated Spark code from Lemonade Workflow
(c) Speed Labs - Departamento de Ciência da Computação
    Universidade Federal de Minas Gerais
More information about Lemonade to be provided.
"""
import traceback
import sys
import os
from pyspark.sql import types # type: ignore
from juicer.util import dataframe_util
from termcolor import colored
from gettext import gettext
import functools

def read_data_iris(spark_session, context):
    """ Read input data."""
    schema_df = types.StructType()
    schema_df.add('sepal_length', types.DecimalType(5, 1), False)
    schema_df.add('sepal_width', types.DecimalType(5, 1), False)
    schema_df.add('petal_length', types.DecimalType(5, 1), False)
    schema_df.add('petal_width', types.DecimalType(5, 1), False)
    schema_df.add('species', types.StringType(), False)

    url = '/'.join(['hdfs:', '', 'spark01.ctweb.inweb.org.br:9000', 'limonero', 'data', 'development', 'iris.data.csv']) #protect 'Protected, please update'
    jvm = spark_session._jvm
    jvm.java.lang.System.setProperty("HADOOP_USER_NAME", "hadoop")
    os.environ['HADOOP_USER_NAME'] = 'hadoop'

    df = spark_session.read.option('nullValue', '').option(
        'treatEmptyValuesAsNulls', 'true').option(
        'wholeFile', True).option(
            'multiLine', True).option('escape',
                '"').option('timestampFormat', 'yyyy/MM/dd HH:mm:ss'
                ).csv(
            url, schema=schema_df,
            quote=None,
            ignoreTrailingWhiteSpace=True, # Handles 
            encoding='UTF-8',
            header=True, sep=',',
            inferSchema=False,
            mode='PERMISSIVE')
    df.cache()
    return df

def execute_cell_1(spark_session, context, emit):
    """ Execute the query. """
    df = None
    success = True
    sql = """
        select *, 'Formata=valor',  '2024-08-23', 12334, '2024-08-01 11:33:53', 
           'Ref format=${ref|%d-%m-%YT%H:%M:%S}', '${date|%d-%m-%YT%H:%M:%S}', ${ref|%Y} from iris
    """.format(**context)
    df = spark_session.sql(sql)
    return (df, success)

def main(spark_session: any, cached_state: dict, emit_event: callable):
    """ Run generated code """
    context = {}

    try:
        # TODO: Evaluate other approach to register Java's UDF
        spark_session.udf.registerJavaFunction(
            "validate_codes",
            "br.ufmg.dcc.lemonade.udfs.ValidateCodesUDF",
            types.BooleanType())
        spark_session.udf.registerJavaFunction(
            "date_patterning",
            "br.ufmg.dcc.lemonade.udfs.DatePatterningUDF",
            types.StringType())
        spark_session.udf.registerJavaFunction(
            "physical_or_legal_person",
            "br.ufmg.dcc.lemonade.udfs.PhysicalOrLegalPersonUDF",
            types.StringType())
        spark_session.udf.registerJavaFunction(
            "strip_accents",
            "br.ufmg.dcc.lemonade.udfs.StripAccentsUDF",
            types.StringType())
    except Exception as e:
        traceback.print_exc(file=sys.stderr)

    try:
        # Keep the names of valid temporary tables only
        tables_to_keep = [
            '`iris`',
            '`sql1`',
        ]
        all_temp_tables = spark_session.catalog.listTables()
        # Filter out the tables that are not in the list to keep
        tables_to_drop = [table.name for table in all_temp_tables if
                          table.name not in tables_to_keep]

        # Drop the unwanted temporary tables
        for table_name in tables_to_drop:
            spark_session.catalog.dropTempView(table_name)
         # Data source 1
        emit_event('update task', message=_('Task running'),
            status='RUNNING', identifier='6c948f7a-cba2-4218-ad45-37bc1f3d9588')
        (hash, result) = cached_state.get(
            '6c948f7a-cba2-4218-ad45-37bc1f3d9588', (None, None))
        if hash == '37444e63907d968b4a4947cb38ce9c019e6b6310':
            df_0 = result
            print(colored('Using cache', 'green'))
        else:
            result = read_data_iris(spark_session, context)
            cached_state['6c948f7a-cba2-4218-ad45-37bc1f3d9588'] = (
                '37444e63907d968b4a4947cb38ce9c019e6b6310', result)
            df_0 = result
        df_0.createOrReplaceTempView('`iris`')
        emit_event('update task', message=_('Task completed'),
            status='COMPLETED', identifier='6c948f7a-cba2-4218-ad45-37bc1f3d9588')

        results = []
        emit_event('update task', message=_('Task running'),
            status='RUNNING', identifier='b10461ec-cc57-49e0-8cf2-079c2f8da5c1')
        # Cell 1
        (hash, result) = cached_state.get(
            'b10461ec-cc57-49e0-8cf2-079c2f8da5c1', (None, None))
        if hash == 'b594844f0727f87e47b378dcd6221cf493236b9b':
            result_1 = result
            df = result
            print(colored('Using cache', 'green'))
        else:
            df, success = execute_cell_1(spark_session, context, emit_event)
            if df is not None:
                cached_state['b10461ec-cc57-49e0-8cf2-079c2f8da5c1'] = (
                    'b594844f0727f87e47b378dcd6221cf493236b9b', df)
                result_1 = df
            else:
                result_1 = None
        results.append([df, 'b10461ec-cc57-49e0-8cf2-079c2f8da5c1'])
        if df is not None:
            result_1.createOrReplaceTempView('`sql1`')
        emit_event('update task', message=_('Task completed'),
            status='COMPLETED', identifier='b10461ec-cc57-49e0-8cf2-079c2f8da5c1')

        if results and results[-1][0].columns:
            task_id = results[-1][1]
            dataframe_util.emit_sample_data_explorer(
                task_id, results[-1][0], emit_event, 'output')

        return cached_state

    except Exception as e:
        spark_session.sparkContext.cancelAllJobs()
        traceback.print_exc(file=sys.stderr)
        if not dataframe_util.handle_spark_exception(e):
            raise