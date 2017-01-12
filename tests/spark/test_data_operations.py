# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

from juicer.spark.data_operation import DataReader
from tests import compare_ast


def test_data_reader_minimal_parameters_no_attributes_success():
    parameters = {
        'data_source': 1
    }
    instance = DataReader(parameters, inputs=[], outputs=['output_1'],
                          named_inputs={}, named_outputs={})
    code = instance.generate_code()
    generated_tree = ast.parse(code)

    expected_tree = ast.parse(dedent("""
        schema_{output} = StructType()
        url_output_1 = '{url}'
        {output} = spark_session.read\
                               .option('nullValue', '')\
                               .option('treatEmptyValuesAsNulls', 'true')\
                               .csv(url_output_1, schema=schema_output_1,
                                    header=False, sep=',',
                                    inferSchema=False, mode='DROPMALFORMED')
        {output}.cache()
        """.format(url='http://hdfs.lemonade:9000', output='output_1')))
    result, msg = compare_ast(generated_tree, expected_tree)
    assert result, msg
    # assert code == "output_1 = spark.read.csv('file', header=True, sep=',')"
