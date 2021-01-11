import datetime
import base64
import random
import hashlib
import pandas as pd
import numpy as np
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import TransformationOperation

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
    """ Test if all functions supported in Lemonade are mapped in scikit-learn
    transpiler.
    """
    now = datetime.datetime.now()
    all_functions = [
        ##("abs", "sepalwidth", [], None),
        ##("acos", "sepalwidth", [], None),
        #("add_months", "sepalwidth", [], None),
        ("array_contains", "array_value", [5],
            lambda row: (5 in row['array_value']) != (
                row['array_contains_new'])
         ),
        ("array_distinct", "array_value", [],
            lambda row: (np.unique(row['array_value']).ravel()
                         != row['array_distinct_new']).all()
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
        ("md5", "class", [256],
            lambda row: row['md5_new'] != hashlib.md5(
                row['class'].encode()).hexdigest()
         ),
        ("minute", "date", [], None),
        #("monotonically_increasing_id", "sepalwidth", [], None),
        ("month", "date", [], None),
        #("months_between", "sepalwidth", [], None),
        #("nanvl", "sepalwidth", [], None),
        ("next_day", "date", ["TU"],
            lambda row: row['next_day_new'] != row['date'] + \
         datetime.timedelta(days=(1-row['date'].weekday()+7) % 7)
         ),
         ("next_day", "date", ["Sunday"],
            lambda row: row['next_day_new'] != row['date'] + \
         datetime.timedelta(days=(6-row['date'].weekday()+7) % 7)
         ),
        # Difficult
        #("posexplode", "sepalwidth", [], None),
        ("pow", "sepalwidth", [12], None),
        ("quarter", "date", [], None),
        ("radians", "sepalwidth", [], None),
        ("rand", "int_value", [], None),
        ("randn", "int_value", [], None),
        ("regexp_extract", "class", [r'\w{2}', 2],
            # ['Ir', 'is', 'se', 'to']
            lambda row: row['regexp_extract_new'] != 'se'
         ),
        ("regexp_replace", "class", ['set', 'SET'],
            lambda row: row['regexp_replace_new'] != row['class'].replace(
                'set', 'SET')
         ),
        ("regexp_replace", "class", [r'[^\w]', '##'],
            lambda row: row['regexp_replace_new'] != row['class'].replace(
                '-', '##')
         ),
        ("repeat", "class", [2],
            lambda row: row['repeat_new'] != row['class'] * 2
         ),
        ("repeat", None, ['abc', 2],
            lambda row: row['repeat_new'] != 'abc' * 2
         ),
        ("replace", "class", ['ri', 'si'],
            lambda row: row['replace_new'] != row['class'].replace('ri', 'si')
         ),
        ("replace", "class", ['ri'],
            lambda row: row['replace_new'] != row['class'].replace('ri', '')
         ),
        ("reverse", "array_value", [],
            lambda row: row['reverse_new'] != row['array_value'][::-1]
         ),
        ("rint", "sepalwidth", [], None),
        ("round", "sepalwidth", [], None),
        ("rpad", "class", [10, '@'], None),
        ("rtrim", "class", [], None),
        #("schema_of_json", "sepalwidth", [], None),
        ("second", "date", [], None),
        ("sequence", None, [1, 10],
            lambda row: row['sequence_new'] != list(range(1, 11))
         ),
        ("sha1", "class", [],
            lambda row: row['sha1_new'] != hashlib.sha1(
                row['class'].encode('utf8')).hexdigest()
         ),
        ("sha2", "class", [256],
            lambda row: row['sha2_new'] != hashlib.sha256(
                row['class'].encode('utf8')).hexdigest()
         ),
        ("sha2", "class", [384],
            lambda row: row['sha2_new'] != hashlib.sha384(
                row['class'].encode('utf8')).hexdigest()
         ),
        #("shiftLeft", "sepalwidth", [], None),
        #("shiftRight", "sepalwidth", [], None),
        #("shiftRightUnsigned", "sepalwidth", [], None),
        ("shuffle", "array_value", [], None),
        ("signum", "sepalwidth", [], None),
        ("sin", "sepalwidth", [], None),
        ("sinh", "sepalwidth", [], None),
        ("size", "array_value", [],
            lambda row: row['size_new'] != len(row['array_value'])
         ),
        ("slice", "array_value", [2, 3],
            lambda row: row['slice_new'] != row['array_value'][1:4]
         ),
        ("slice", "array_value", [-2, 3],
            lambda row: row['slice_new'] != row['array_value'][-2:1]
         ),
        ("sort_array", "array_value", [],
            None,
            # lambda row: row['sort_array_new'] != sorted('array_value')
         ),
        ("sort_array", "array_value", [True], None
            # lambda row: row['sort_array_new'] != sorted('array_value',
            #                                             reverse=True)
         ),
        #("soundex", "sepalwidth", [], None),
        ("split", "class", [r'\W'],
            lambda row: row['split_new'] != row['class'].split('-')
         ),
        ("split", "class", [r'[aeiou]'],
            lambda row: len(row['split_new']) != 5
         ),
        ("sqrt", "sepalwidth", [], None),
        #("struct", "sepalwidth", [], None),
        ("substring", "class", [2, 4],
            lambda row: row['substring_new'] != row['class'][2:4]
         ),
        ("substring", "class", [2],
            lambda row: row['substring_new'] != row['class'][2:]
         ),
        ("substring_index", "class", ['i', 2],
            lambda row: row['substring_index_new'] != row['class'][2:]
         ),
        ("tan", "sepalwidth", [], None),
        ("tanh", "sepalwidth", [], None),
        ("to_date", "date_str", ['yyyy-MM-dd HH:mm:ss'], None),
        ("to_json", "sepalwidth", [], None),
        ("to_timestamp", "date_str", ['yyyy-MM-dd HH:mm:ss'], None),
        ("to_utc_timestamp", "date", ['America/Santiago'],
            lambda row: row['to_utc_timestamp_new'].tzinfo.zone !=
         'America/Santiago'
         ),
        ("to_utc_timestamp", "date", ['Europe/Berlin'],
            lambda row:
                row['to_utc_timestamp_new'].tzinfo.zone != 'Europe/Berlin'
         ),
        ("translate", "class", ['ri', 'XY'],
            lambda row: row['translate_new'] != row['class'].replace(
                'ri', 'XY')),
        ("trim", "class", [], None),
        ("trunc", "date", ['yyyy'],
            lambda row: row['trunc_new'] != datetime.date.today().replace(
                month=1, day=1)),
        ("trunc", "date", ['yyyy'],
            lambda row: row['trunc_new'] != datetime.date.today().replace(
                day=1)),
        ("trunc", "date", ['mm'],
            lambda row: row['trunc_new'] != datetime.date.today().replace(
                day=1)),
        ("unbase64", "base64", [], None),
        ("unhex", "hex", [], None),
        ("unix_timestamp", "date_str", [], None),
        ("upper", "class", [], None),
    ]
    to_test = ['next_day']
    dd = ['df', util.iris(['sepalwidth',
                           'petalwidth', 'class'], 10)]

    df = dd[1]

    df['date_str'] = pd.Series(['2020-01-01 12:32:12' for x in range(10)])
    df['date_str2'] = pd.Series(['12-11-21' for x in range(10)])
    df['date'] = pd.Series([now for x in range(10)])
    df['hex'] = pd.Series(['Fa03' for x in range(10)])
    df['base64'] = pd.Series(
        [base64.b64encode('test lemonade'.encode('utf8')) for x in range(10)])
    df['int_value'] = pd.Series([888 for x in range(10)])
    df['array_value'] = pd.Series(
        [[random.randrange(1, 10, 1) for i in range(5)] for x in range(10)])

    test_functions = [x for x in all_functions if x[0] in to_test]
    for func_name, field, args, test in test_functions:
        print('Testing', func_name)
        expr = {
            "alias": "{}_new".format(func_name),
            "expression": "{}(attr0)".format(func_name),
            "tree": {
                "type": "CallExpression",
                "arguments": [
                        {"type": "Identifier", "name": field}
                ] if field else [],
                "callee": {"type": "Identifier", "name": "{}".format(func_name)}
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
                'expression': [expr],
            },
            'named_inputs': {
                'input data': dd[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        instance = TransformationOperation(**arguments)
        code = instance.generate_code()
        util.execute(code, dict([dd]))
        if test is not None:
            import pdb
            pdb.set_trace()
            assert len(df[df.apply(test, axis=1)]) == 0
            # assert test(
        # assert result['out'].equals(util.iris(size=slice_size))
