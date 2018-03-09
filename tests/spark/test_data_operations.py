# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

from juicer.spark.data_operation import DataReaderOperation
from tests import compare_ast, format_code_comparison


def test_data_reader_minimal_parameters_no_attributes_success():
    parameters = {
        'data_source': 1,
        'configuration': {
            'juicer': {
                'services': {
                    'limonero': {
                        'url': 'http://limonero:12345',
                        'auth_token': 'zzzz'
                    }
                }
            }
        },
        'workflow': {'data_source_cache': {}}
    }
    n_out = {'output data': 'output_1'}

    instance = DataReaderOperation(parameters, named_inputs={}, named_outputs=n_out)
    code = instance.generate_code()
    generated_tree = ast.parse(code)

    expected_code = dedent("""
        schema_{output} = types.StructType()
        schema_{output}.add('value', types.StringType(), 1, None)
        url = '{url}'
        {output} = spark_session.read\\
                               .option('nullValue', '')\\
                               .option('treatEmptyValuesAsNulls', 'true')\\
                               .option('wholeFile', True)\\
                               .option('multiLine', True)\\
                               .option('escape', '"')\\
                               .csv(url, schema=schema_output_1,
                                    quote=None, encoding='UTF-8',
                                    header=False, sep=',',
                                    inferSchema=False, mode='FAILFAST')
        {output}.cache()
        """.format(url='http://hdfs.lemonade:9000', output='output_1'))
    expected_tree = ast.parse(expected_code)
    result, msg = compare_ast(generated_tree, expected_tree)
    assert result, msg + format_code_comparison(code, expected_code)
    # assert code == "output_1 = spark.read.csv('file', header=True, sep=',')"
