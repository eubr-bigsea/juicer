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
        ]
        all_temp_tables = spark_session.catalog.listTables()
        # Filter out the tables that are not in the list to keep
        tables_to_drop = [table.name for table in all_temp_tables if
                          table.name not in tables_to_keep]

        # Drop the unwanted temporary tables
        for table_name in tables_to_drop:
            spark_session.catalog.dropTempView(table_name)

        results = []

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