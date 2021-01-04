from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import TransformationOperation
import pytest
import datetime
import json
import pandas as pd
import numpy as np
import base64
import random

# Transformation
# 
# def test_transformation_success():
#     slice_size = 10
#     df = ['df', util.iris(['sepallength', 'sepalwidth', 
#         'petalwidth', 'petallength'], slice_size)]
# 
#     arguments = {
#         'parameters': {},
#         'named_inputs': {
#             'input data': df[0],
#         },
#         'named_outputs': {
#             'output data': 'out'
#         }
#     }
#     instance = TransformationOperation(**arguments)
#     result = util.execute(instance.generate_code(), 
#                           dict([df]))
#     assert result['out'].equals(util.iris(size=slice_size))

def test_function_implementation():
    """ Test if all functions supported in Lemonade are 
    mapped in scikit-learn transpiler. 
    """
    all_functions = [
        ##("abs", "sepalwidth", [], None),
        ##("acos", "sepalwidth", [], None),
        #("add_months", "sepalwidth", [], None),
        ("array_contains", "array_value", [5], 
            lambda row: (5 in row['array_value']) != (row['array_contains_new'])
        ),
        ("array_distinct", "array_value", [], 
            lambda row: (np.unique(row['array_value']).ravel() != row['array_distinct_new']).all()
        ),
        #("array_except", "array_value", [], None),
        #("array_intersect", "array_value", [], None),
        #("array_join", "array_value", [], None),
        #("array_max", "array_value", [], None),
        #("array_min", "array_value", [], None),
        #("array_position", "array_value", [], None),
        #("array_remove", "array_value", [], None),
        #("array_repeat", "array_value", [], None),
        #("array_sort", "array_value", [], None),
        #("array_union", "array_value", [], None),
        #("arrays_overlap", "array_value", [], None),
        #("arrays_zip", "array_value", [], None),
        ("ascii", "class", [], None),
        ("asin", "sepalwidth", [], None),
        ("atan", "sepalwidth", [], None),
        ("atan2", "sepalwidth", [0], None),
        #("base64", "sepalwidth", [], None),
        #("basestring", "sepalwidth", [], None),
        ("bin", "int_value", [], None), 
        #("bitwiseNOT", "sepalwidth", [], None),
        #("bround", "sepalwidth", [], None),
        ("cbrt", "sepalwidth", [], None),
        ("ceil", "sepalwidth", [], None),
        #("coalesce", "sepalwidth", [], None),
        #("concat", "sepalwidth", [], None),
        #("concat_ws", "sepalwidth", [], None),
        #("conv", "sepalwidth", [], None),
        ("cos", "sepalwidth", [], None),
        ("cosh", "sepalwidth", [], None),
        #("crc32", "sepalwidth", [], None),
        #("create_map", "sepalwidth", [], None),
        ("current_date", "sepalwidth", [], None),
        ("current_timestamp", "sepalwidth", [], None),
        #("date_add", "sepalwidth", [], None),
        #("date_format", "sepalwidth", [], None),
        #("date_sub", "sepalwidth", [], None),
        #("date_trunc", "sepalwidth", [], None),
        #("datediff", "date", ['2009-07-30'], None),
        ("dayofmonth", "date", [], None),
        ("dayofweek", "date", [], None),
        ("dayofyear", "date", [], None),
        #("decode", "sepalwidth", [], None),
        ("degrees", "sepalwidth", [], None),
        #("element_at", "sepalwidth", [], None),
        #("encode", "sepalwidth", [], None),
        ("exp", "sepalwidth", [], None),
        #("explode", "sepalwidth", [], None),
        #("explode_outer", "sepalwidth", [], None),
        ("expm1", "sepalwidth", [], None),
        #("flatten", "sepalwidth", [], None),
        ("floor", "sepalwidth", [], None),
        #("format_number", "sepalwidth", [], None),
        #("from_json", "sepalwidth", [], None),
        #("from_unixtime", "sepalwidth", [], None),
        #("from_utc_timestamp", "sepalwidth", [], None),
        #("get_json_object", "sepalwidth", [], None),
        #("greatest", "sepalwidth", [], None),
        #("hash", "int_value", [], None),
        ("hex", "int_value", [], None),
        ("hour", "date", [], None),
        ("hypot", "sepalwidth", [34], None),
        ("initcap", "class", [], None),
        ("instr", "class", ['iris'], None),
        ("isnan", "sepalwidth", [], None),
        #("isnull", "sepalwidth", [], None),
        #("json_tuple", "sepalwidth", [], None),
        #("last_day", "date", [], None),
        #("least", "sepalwidth", [0, 200, 'None'], None),
        ("length", "class", [], None),
        #("levenshtein", "sepalwidth", [], None),
        ("lit", "sepalwidth", [], None),
        #("locate", "class", ['iris', 10], None),
        ("log", "sepalwidth", [], None),
        ("log10", "sepalwidth", [], None),
        ("log1p", "sepalwidth", [], None),
        ("log2", "sepalwidth", [], None),
        ("lower", "class", [], None),
        ("lpad", "class", [10, '@'], None),
        ("ltrim", "class", [], None),
        #("map_concat", "sepalwidth", [], None),
        #("map_from_arrays", "sepalwidth", [], None),
        #("map_keys", "sepalwidth", [], None),
        #("map_values", "sepalwidth", [], None),
        #("md5", "sepalwidth", [], None),
        ("minute", "date", [], None),
        #("monotonically_increasing_id", "sepalwidth", [], None),
        ("month", "date", [], None),
        #("months_between", "sepalwidth", [], None),
        #("nanvl", "sepalwidth", [], None),
        #("next_day", "sepalwidth", [], None),
        #("posexplode", "sepalwidth", [], None),
        ("pow", "sepalwidth", [12], None),
        ("quarter", "date", [], None),
        ("radians", "sepalwidth", [], None),
        ("rand", "int_value", [], None),
        ("randn", "int_value", [], None),
        #("regexp_extract", "sepalwidth", [], None),
        # ("regexp_replace", "sepalwidth", [], None),
        # ("repeat", "sepalwidth", [], None),
        ("reverse", "class", [], None),
        ("rint", "sepalwidth", [], None),
        ("round", "sepalwidth", [], None),
        ("rpad", "class", [10, '@'], None),
        ("rtrim", "class", [], None),
        #("schema_of_json", "sepalwidth", [], None),
        ("second", "date", [], None),
        #("sequence", "sepalwidth", [], None),
        #("sha1", "sepalwidth", [], None),
        #("sha2", "sepalwidth", [], None),
        #("shiftLeft", "sepalwidth", [], None),
        #("shiftRight", "sepalwidth", [], None),
        #("shiftRightUnsigned", "sepalwidth", [], None),
        #("shuffle", "sepalwidth", [], None),
        ("signum", "sepalwidth", [], None),
        ("sin", "sepalwidth", [], None),
        ("sinh", "sepalwidth", [], None),
        #("size", "sepalwidth", [], None),
        #("slice", "sepalwidth", [], None),
        #("sort_array", "sepalwidth", [], None),
        #("soundex", "sepalwidth", [], None),
        #("split", "class", [], None),
        ("sqrt", "sepalwidth", [], None),
        #("struct", "sepalwidth", [], None),
        #("substring", "class", [], None),
        #("substring_index", "sepalwidth", [], None),
        ("tan", "sepalwidth", [], None),
        ("tanh", "sepalwidth", [], None),
        #("to_date", "sepalwidth", [], None),
        ("to_json", "sepalwidth", [], None),
        #("to_timestamp", "sepalwidth", [], None),
        #("to_utc_timestamp", "date", [], None),
        # ("translate", "sepalwidth", [], None),
        ("trim", "class", [], None),
        ("trunc", "sepalwidth", [], None),
        ("unbase64", "base64", [], None),
        ("unhex", "hex", [], None),
        ("unix_timestamp", "date_str", [], None),
        ("upper", "class", [], None),
    ]
    dd = ['df', util.iris(['sepalwidth', 
         'petalwidth', 'class'], 10)]

    df = dd[1]

    df['date_str'] = pd.Series(['2020-01-01 12:32:12' for x in range(10)]) 
    df['date'] = pd.Series([datetime.datetime.now() for x in range(10)]) 
    df['hex'] = pd.Series(['Fa03' for x in range(10)]) 
    df['base64'] = pd.Series([base64.b64encode('test lemonade'.encode('utf8')) for x in range(10)]) 
    df['int_value'] = pd.Series([888 for x in range(10)]) 
    df['array_value'] = pd.Series([[random.randrange(1, 10, 1) for i in range(5)] for x in range(10)]) 

    for f, field, args, test in all_functions[:10]:
        expr = {
                "alias":"{}_new".format(f),
                "expression":"{}(attr0)".format(f),
                "tree":{
                    "type":"CallExpression",
                    "arguments":[
                        {"type":"Identifier","name": field}
                    ],
                    "callee":{"type":"Identifier","name":"{}".format(f)}
                }
        }
        if args:
            expr['tree']['arguments'] += [
                {'type': 'Literal', 'value': v, 'raw': v} 
                for v in args
            ]
        arguments = {
            'parameters': {
                'multiplicity': {'input data': 1},
                'expression': [expr]
            },
            'named_inputs': {
                'input data': dd[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        }   
        instance = TransformationOperation(**arguments)
        result = util.execute(instance.generate_code(), dict([dd]))
        if test is not None:
            #import pdb
            #pdb.set_trace()
            assert len(df[df.apply(test, axis=1)]) == 0
            #assert test( 
        # assert result['out'].equals(util.iris(size=slice_size))

