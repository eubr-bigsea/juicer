# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

import mock
from juicer.spark.data_operation import DataReaderOperation
from tests import compare_ast, format_code_comparison


def mock_query_limonero(base_url, item_path, token, item_id):
    result = {"id": item_id, "name": "test.csv",
              "description": "Test", "enabled": True,
              "statistics_process_counter": 0, "read_only": True,
              "privacy_aware": False,
              "url": "hdfs://server/data/development/test.csv",
              "created": "2017-07-20T17:14:20+00:00",
              "updated": "2018-01-01T00:00:00+00:00", "format": "CSV",
              "user_id": 1, "user_login": "limonero",
              "user_name": "bigsea", "temporary": False,
              "workflow_id": None, "task_id": None, "attribute_delimiter": None,
              "record_delimiter": None, "text_delimiter": None,
              "is_public": False, "treat_as_missing": None, "encoding": None,
              "is_first_line_header": False, "is_multiline": False,
              "command": None,
              "attributes": [
                  {"id": 213, "name": "id", "description": None,
                   "type": "CHARACTER", "size": 7, "precision": None,
                   "scale": None,
                   "nullable": False, "enumeration": False,
                   "missing_representation": None, "feature": False,
                   "label": False,
                   "distinct_values": None, "mean_value": None,
                   "median_value": None,
                   "max_value": None, "min_value": None, "std_deviation": None,
                   "missing_total": None, "deciles": None,
                   "attribute_privacy": None},
                  {"id": 214, "name": "name", "description": None,
                   "type": "CHARACTER", "size": 8, "precision": None,
                   "scale": None,
                   "nullable": False, "enumeration": False,
                   "missing_representation": None, "feature": False,
                   "label": False,
                   "distinct_values": None, "mean_value": None,
                   "median_value": None,
                   "max_value": None, "min_value": None, "std_deviation": None,
                   "missing_total": None, "deciles": None,
                   "attribute_privacy": None},
                  {"id": 215, "name": "year", "description": None,
                   "type": "CHARACTER", "size": 4, "precision": None,
                   "scale": None,
                   "nullable": False, "enumeration": False,
                   "missing_representation": None, "feature": False,
                   "label": False,
                   "distinct_values": None, "mean_value": None,
                   "median_value": None,
                   "max_value": None, "min_value": None, "std_deviation": None,
                   "missing_total": None, "deciles": None,
                   "attribute_privacy": None}], "permissions": [],
              "storage": {"id": 1, "name": "Default", "type": "HDFS",
                          "enabled": True,
                          "url": "hdfs://server:9000"}}
    return result


def test_data_reader_minimal_parameters_success():
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

    with mock.patch('juicer.service.limonero_service.query_limonero',
                    mock_query_limonero) as mocked_limonero:
        instance = DataReaderOperation(parameters, named_inputs={},
                                       named_outputs=n_out)

        code = instance.generate_code()
    generated_tree = ast.parse(code)
    url = 'hdfs://server/data/development/test.csv'
    expected_code = dedent("""
        schema_{output} = types.StructType()
        schema_output_1.add('id', types.StringType(), False)
        schema_output_1.add('name', types.StringType(), False)
        schema_output_1.add('year', types.StringType(), False)

        url = '{url}'
        {output} = spark_session.read\\
                       .option('nullValue', '')\\
                       .option('treatEmptyValuesAsNulls', 'true')\\
                       .option('wholeFile', True)\\
                       .option('multiLine', True)\\
                       .option('escape', '"')\\
                       .option('timestampFormat', 'yyyy/MM/dd HH:mm:ss')\\
                       .csv(url, schema=schema_output_1,
                            quote=None, encoding='UTF-8',
                            header=False, sep=',',
                            inferSchema=False, mode='FAILFAST')
        {output}.cache()
        """.format(url=url, output='output_1'))
    expected_tree = ast.parse(expected_code)
    result, msg = compare_ast(generated_tree, expected_tree)
    assert result, msg + format_code_comparison(code, expected_code)
    # assert code == "output_1 = spark.read.csv('file', header=True, sep=',')"
