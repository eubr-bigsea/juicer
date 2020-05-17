# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

import pytest
from tests import compare_ast, format_code_comparison
from juicer.scikit_learn.etl_operation import AddColumnsOperation, \
    AggregationOperation, CleanMissingOperation, \
    DifferenceOperation, DistinctOperation, DropOperation, \
    ExecutePythonOperation, ExecuteSQLOperation, \
    FilterOperation, IntersectionOperation, JoinOperation, \
    ReplaceValuesOperation,  SampleOrPartitionOperation, \
    SelectOperation, SortOperation, SplitOperation, \
    TransformationOperation,  UnionOperation


def debug_ast(code, expected_code):
    print("""
    Code
    {sep}
    {code}
    {sep}
    Expected
    {sep}
    {expected}
    """.format(code=code, sep='-' * 20, expected=expected_code))


'''
    Add Columns Operation
'''


def test_add_columns_minimum_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = AddColumnsOperation(parameters=params,
                                   named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    out = pd.merge(df1, df2, left_index=True, 
        right_index=True, suffixes=('_ds0', '_ds1'))
    """.format(
        out=n_out['output data'],
        in0=n_in['input data 1'],
        in1=n_in['input data 2']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_add_columns_suffixes_params_success():
    params = {AddColumnsOperation.ALIASES_PARAM: '_l,_r'}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = AddColumnsOperation(parameters=params,
                                   named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    out = pd.merge(df1, df2, left_index=True, 
        right_index=True, suffixes=('_l', '_r'))
    """.format(
        out=n_out['output data'],
        in0=n_in['input data 1'],
        in1=n_in['input data 2']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


'''
    Clean Missing Operation
'''


def test_clean_missing_minimal_params_success():
    params = {
        CleanMissingOperation.ATTRIBUTES_PARAM: ['col1', 'col2'],
        CleanMissingOperation.MIN_MISSING_RATIO_PARAM: 0.0,
        CleanMissingOperation.MAX_MISSING_RATIO_PARAM: 1.0,
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output result': 'output_1'}
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
     min_missing_ratio = 0.0
     max_missing_ratio = 1.0
     output_1 = input_1
     for col in ['col1', 'col2']:
        ratio = input_1[col].isnull().sum()
        if ratio >= min_missing_ratio and ratio <= max_missing_ratio:
            output_1.dropna(subset=col, axis='index', inplace=True)
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_value_param_failure():
    params = {
        CleanMissingOperation.CLEANING_MODE_PARAM: "VALUE"
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    with pytest.raises(ValueError):
        CleanMissingOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_clean_wrong_ratio_param_failure():
    params = {
        CleanMissingOperation.MIN_MISSING_RATIO_PARAM: 1.7,
        CleanMissingOperation.MAX_MISSING_RATIO_PARAM: -1.0,
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    with pytest.raises(ValueError):
        CleanMissingOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_clean_missing_without_missing_rating_params_success():
    params = {
        CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output result': 'output_1'}
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
        min_missing_ratio = 0.0
        max_missing_ratio = 1.0
        {output_1} = {input_1}
        for col in {attribute}:
            ratio = {input_1}[col].isnull().sum()
            if ratio >= min_missing_ratio and ratio <= max_missing_ratio:
                {output_1}.dropna(subset=col, axis='index', inplace=True)
    """.format(input_1=n_in['input data'], attribute=params['attributes'],
               output_1=n_out['output result']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_minimal_params_type_value_success():
    params = {
        CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
        CleanMissingOperation.MIN_MISSING_RATIO_PARAM: 0.0,
        CleanMissingOperation.MAX_MISSING_RATIO_PARAM: 1.0,
        CleanMissingOperation.VALUE_PARAMETER: 200,
        CleanMissingOperation.CLEANING_MODE_PARAM: 'VALUE'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output result': 'output_1'}
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
        min_missing_ratio = 0.0
        max_missing_ratio = 1.0
        output_1 = input_1
        for col in ['name']:
            ratio = input_1[col].isnull().sum()
            if ratio >= min_missing_ratio and ratio <= max_missing_ratio:
                output_1[col].fillna(value=200, inplace=True)
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)

    # Test with value being number
    params[CleanMissingOperation.VALUE_PARAMETER] = 1200
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = expected_code.replace('200', '1200')
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


'''
    Difference Operation
'''


def test_difference_minimal_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    class_name = DifferenceOperation
    instance = class_name(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = "cols = {in1}.columns\n" \
                    "{out} = pd.merge({in1}, {in2}, " \
                    "indicator=True, how='left', on=None)\n" \
                    "{out} = {out}.loc[{out}['_merge'] == 'left_only', cols]"\
        .format(out=n_out['output data'], in1=n_in['input data 1'],
                in2=n_in['input data 2'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


'''
    Distinct Operation
'''


def test_remove_duplicated_minimal_params_success():
    params = {}
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = DistinctOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {in1}.drop_duplicates(subset=None, keep='first')"\
        .format(out=n_out['output data'], in1=n_in['input data'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_duplicated_by_attributes_success():
    params = {
        'attributes': ['name']
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = DistinctOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{} = {}.drop_duplicates(subset={}, keep='first')"\
        .format(n_out['output data'], n_in['input data'], params['attributes'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


'''
    Drop Operation
'''


def test_drop_minimal_params_success():
    params = {
        'attributes': 'TEST'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = DropOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {input}.drop(columns={columns})" \
        .format(out=n_out['output data'], input=n_in['input data'],
                columns=params['attributes'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


'''
    Execute Python Operation
'''


def test_pythoncode_minimum_params_success():
    params = {
        ExecutePythonOperation.PYTHON_CODE_PARAM:
            "df1['col3'] =  df1['col1'] + df1['col2']",
        'task': {'id': 1}
    }
    n_in = {'input data 1': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = ExecutePythonOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    import json                                                                            
    from RestrictedPython.Guards import safe_builtins                                      
    from RestrictedPython.RCompile import compile_restricted                               
    from RestrictedPython.PrintCollector import PrintCollector                             
                                                                                           
    results = [r[1].result() for r in task_futures.items() if r[1].done()]                 
    results = dict([(r['task_name'], r) for r in results])                                 
    # Input data                                                                           
    in1 = input_1                                                                          
    in2 = None                                                                             
                                                                                           
    # Output data, initialized as None                                                     
    out1 = None                                                                            
    out2 = None                                                                            
                                                                                           
    # Variables and language supported                                                     
    ctx = {                                                                                
        'wf_results': results,                                                             
        'in1': in1,                                                                        
        'in2': in2,                                                                        
        'out1': out1,                                                                      
        'out2': out2,                                                                      
                                                                                           
        # Restrictions in Python language                                                  
         '_write_': lambda v: v,                                                           
        '_getattr_': getattr,                                                              
        '_getitem_': lambda ob, index: ob[index],                                          
        '_getiter_': lambda it: it,                                                        
        '_print_': PrintCollector,                                                         
        'json': json,                                                                      
    }                                                                                      
    user_code = "df1['col3'] =  df1['col1'] + df1['col2']"   
                                                                                           
    ctx['__builtins__'] = safe_builtins                                                    
                                                                                           
    compiled_code = compile_restricted(user_code,                                          
    str('python_execute_1'), str('exec'))                                                  
    try:                                                                                   
        exec compiled_code in ctx                                                          
                                                                                           
        # Retrieve values changed in the context                                           
        out1 = ctx['out1']                                                                 
        out2 = ctx['out2']                                                                 
                                                                                           
        if '_print' in ctx:                                                                
            emit_event(name='update task',                                                 
                message=ctx['_print'](),                                                   
                status='RUNNING',                                                          
                identifier='1')                                                            
    except NameError as ne:                                                                
        raise ValueError(_('Invalid name: {}. '                                            
            'Many Python commands are not available in Lemonade').format(ne))              
    except ImportError as ie:                                                              
        raise ValueError(_('Command import is not supported'))                             
                                                                                           
    out_1_1 = out1                                                                         
    out_2_1 = out2                         
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_pythoncode_missing_parameter_failure():
    params = {
        'task': {'id': 1}
    }
    with pytest.raises(ValueError):
        n_in = {'input data 1': 'input_1'}
        n_out = {'output data': 'output_1'}
        ExecutePythonOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Execute SQL Operation
'''


def test_sql_minimum_params_success():
    params = {
        ExecuteSQLOperation.QUERY_PARAM: "select * where df1.id = 1;"
    }
    n_in = {'input data 1': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = ExecuteSQLOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    
        query = 'select * where df1.id = 1;'
        output_1 = sqldf(query, {'ds1': input_1, 'ds2': None})
        names = None

        if names is not None and len(names) > 0:
            old_names = output_1.columns
            if len(old_names) != len(names):
                raise ValueError('Invalid names. Number of attributes '
                                 'in result differs from names informed.')
            rename = dict(zip(old_names, names))
            output_1.rename(columns=rename, inplace=True)
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_sql_two_inputs_params_success():
    params = {
        ExecuteSQLOperation.QUERY_PARAM: "select * where df2.id = 1;",
        ExecuteSQLOperation.NAMES_PARAM: "col1, col2, col3"

    }
    n_in = {'input data 1': 'input_1', 'input data 2': 'input_2'}
    n_out = {'output data': 'output_1'}
    instance = ExecuteSQLOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    
        query = 'select * where df2.id = 1;'
        output_1 = sqldf(query, {'ds1': input_1, 'ds2': input_2})
        names = ['col1', 'col2', 'col3']

        if names is not None and len(names) > 0:
            old_names = output_1.columns
            if len(old_names) != len(names):
                raise ValueError('Invalid names. Number of attributes '
                                 'in result differs from names informed.')
            rename = dict(zip(old_names, names))
            output_1.rename(columns=rename, inplace=True)
        """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_sql_missing_parameter_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data 1': 'input_1'}
        n_out = {'output data': 'output_1'}
        ExecuteSQLOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_wrong_query_parameter_failure():
    params = {
        ExecuteSQLOperation.QUERY_PARAM: "ALTER TABLE Customer DROP Birth_Date;"
    }
    with pytest.raises(ValueError):
        n_in = {'input data 1': 'input_1'}
        n_out = {'output data': 'output_1'}
        ExecuteSQLOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Filter Operation
'''


def test_filter_minimum_params_success():
    params = {
        FilterOperation.FILTER_PARAM: [{
            'attribute': 'code',
            'f': '>',
            'value': '201'
        },
            {
                'attribute': 'code2',
                'f': '<',
                'value': '200'
            }
        ]
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = FilterOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    output_1 = input_1
    output_1 = output_1.query('(code > 201) and (code2 < 200)')
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_filter_advanced_params_success():
    params = {
        FilterOperation.FILTER_PARAM: [{
            'attribute': 'code',
            'f': '>',
            'value': '201'
        }],
        FilterOperation.ADVANCED_FILTER_PARAM: [{
            "alias": "result",
            "expression": "len(col1) == 3",
            "tree": {"operator": "==",
                      "right": {"raw": "3", "type": "Literal", "value": 3},
                      "type": "BinaryExpression",
                     "left": {"type": "CallExpression",
                               "callee": {
                                   "type": "Identifier",
                                   "name": "len"},
                               "arguments": [{
                                                 "type": "Identifier",
                                                 "name": "col1"}]}},
            "error": 'null'}]
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = FilterOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        output_1 = output_1[output_1.apply(lambda row: 
        len(row['col1']) == 3, axis=1)]
        output_1 = output_1.query('(code > 201)')""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_filter_missing_parameter_filter_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FilterOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Intersection Operation
'''


def test_intersection_minimal_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = IntersectionOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent(
        """
        if len({in1}.columns) != len({in2}.columns):
            raise ValueError('{error}')
        {in1} = {in1}.dropna(axis=0, how='any')
        {in2} = {in2}.dropna(axis=0, how='any')
        keys = {in1}.columns.tolist()
        {in1} = pd.merge({in1}, {in2}, how='left', on=keys, 
        indicator=True, copy=False)
        {out} = {in1}.loc[{in1}['_merge'] == 'both', keys]
        """.format(
            out=n_out['output data'], in1=n_in['input data 1'],
            in2=n_in['input data 2'],
            error=(
                'For intersection operation, both input data '
                'sources must have the same number of attributes '
                'and types.')))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


'''
    Join Operation
'''


def test_join_inner_join_minimal_with_remove_right_columns_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id', 'cod'],
        'aliases': '_left,_right'
    }
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        cols_to_remove = [c+'_right' for c in df2.columns if c in df1.columns]

        out = pd.merge(df1, df2, how='inner', suffixes=['_left', '_right'],
                left_on=['id', 'cod'], right_on=['id', 'cod'])

        out.drop(cols_to_remove, axis=1, inplace=True)""")

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_left_join_keep_columns_minimal_params_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id', 'cod'],
        JoinOperation.JOIN_TYPE_PARAM: 'left',
        JoinOperation.KEEP_RIGHT_KEYS_PARAM: True,
        'aliases': '_left,_right'
    }
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        {out} = pd.merge({in0}, {in1}, how='{type}', 
            suffixes=['_left', '_right'], left_on=['id', 'cod'],
            right_on=['id', 'cod'])
        """.format(
            out=n_out['output data'], in0=n_in['input data 1'],
            in1=n_in['input data 2'],
            type=params[JoinOperation.JOIN_TYPE_PARAM], ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_remove_right_with_case_columns_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id2', 'cod2'],
        JoinOperation.KEEP_RIGHT_KEYS_PARAM: False,
        JoinOperation.MATCH_CASE_PARAM: True,
        'aliases': '_left,_right'
    }
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        cols_to_remove = [c+'_right' for c in df2.columns if c in df1.columns]

        data1_tmp = df1[['id', 'cod']].applymap(lambda col: str(col).lower())
        data1_tmp.columns = [c+"_lower" for c in data1_tmp.columns]
        data1_tmp = pd.concat([df1, data1_tmp], axis=1, sort=False)

        data2_tmp = df2[['id2', 'cod2']].applymap(lambda col: str(col).lower())
        data2_tmp.columns = [c+"_lower" for c in data2_tmp.columns]
        data2_tmp = pd.concat([df2, data2_tmp], axis=1, sort=False)

        out = pd.merge(data1_tmp, data2_tmp, left_on=col1, right_on=col2,
            copy=False, suffixes=['_left', '_right'], how='inner')
        out.drop(col1+col2, axis=1, inplace=True)

        out.drop(cols_to_remove, axis=1, inplace=True)""")

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_missing_left_or_right_param_failure():
    params = {
        'right_attributes': ['id', 'cod']
    }
    with pytest.raises(ValueError):
        n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
        n_out = {'output data': 'out'}
        JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    params = {
        'left_attributes': ['id', 'cod']
    }
    with pytest.raises(ValueError):
        JoinOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Replace Values Operation
'''


def test_replace_value_minimal_params_success():
    params = {
        "attributes": ["col1", "col2"],
        "replacement": 10,
        "value": -10
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = ReplaceValuesOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent(
        """
        output_1 = input_1
        replacement = {replaces}
        for col in replacement:
            list_replaces = replacement[col]
            output_1[col] = output_1[col].replace(list_replaces[0],
            list_replaces[1])
        """.format(out=n_out['output data'], in1=n_in['input data'],
                   replaces={"col2": [[-10], [10]], "col1": [[-10], [10]]}))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_replace_value_missing_attribute_param_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        ReplaceValuesOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Sample Operation
'''

def test_sample_or_partition_minimal_params_success():
    params = {
        'fraction': '3',
        'seed': '0'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(frac={}, random_state={})"\
        .format('0.03', params['seed'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_value_success():
    params = {
        'value': '400',
        'seed': '0',
        'type': SampleOrPartitionOperation.TYPE_VALUE
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(n={}, random_state=0)"\
        .format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_head_success():
    params = {
        'value': 365,
        'seed': 0,
        'type': SampleOrPartitionOperation.TYPE_HEAD
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.head({})".format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg


def test_sample_or_partition_invalid_value_failure():
    params = {
        'value': -365,
        'seed': '0',
        'type': SampleOrPartitionOperation.TYPE_HEAD
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


def test_sample_or_partition_invalid_fraction_failure():
    params = {
        'fraction': '101',
        'seed': '0'
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


def test_sample_or_partition_fraction_percentage_success():
    params = {
        'fraction': 45
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(frac={}, random_state={})"\
        .format(params['fraction'] * 0.01, 'None')
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_fraction_missing_failure():
    params = {
        'seed': '0'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


'''
    Select Operation
'''


def test_select_minimal_params_success():
    params = {
        SelectOperation.ATTRIBUTES_PARAM: ['name', 'class']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output projected data': 'output_1'}
    instance = SelectOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    columns = ', '.join(
        ['"{}"'.format(x) for x in params[SelectOperation.ATTRIBUTES_PARAM]])
    expected_code = '{out} = {in1}[[{columns}]]'\
        .format(out=n_out['output projected data'],
                in1=n_in['input data'], columns=columns)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_select_missing_attribute_param_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SelectOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Sort Operation
'''


def test_sort_minimal_params_success():
    params = {
        'attributes': [{'attribute': 'name', 'f': 'asc'},
                       {'attribute': 'class', 'f': 'desc'}],
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = SortOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    code1 = "{out} = {in0}.sort_values(by=['name', 'class'], " \
            "ascending=[True, False])"\
        .format(out=n_out['output data'], in0=n_in['input data'])
    result, msg = compare_ast(ast.parse(code), ast.parse(code1))
    assert result, msg


def test_sort_missing_attributes_failure():
    params = {}
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        SortOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Split Operation
'''


def test_random_split_params_success():
    params = {
        'weights': '40.0',
        'seed': '1234321'
    }
    n_in = {'input data': 'df1'}
    n_out = {'splitted data 1': 'out1', 'splitted data 2': 'out2'}

    instance = SplitOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()

    expected_code = """{out1}, {out2} = np.split({input}.sample(frac=1, 
    random_state={seed}), [int({weights}*len({input}))])
    """.format(out1=n_out['splitted data 1'], out2=n_out['splitted data 2'],
               input=n_in['input data'], weights='0.4', seed=1234321)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


'''
    Transformation Operation
'''


def test_transformation_minumum_params_success():

    params = {
        "expression": [
            {"alias": "new_col1", "expression": "col1+2*9",
             "tree": {"operator": "+", "right": {"operator": "*",
                                                 "right": {"raw": "9",
                                                           "type": "Literal",
                                                           "value": 9},
                                                 "type": "BinaryExpression",
                                                 "left": {"raw": "2",
                                                          "type": "Literal",
                                                          "value": 2}},
                      "type": "BinaryExpression",
                      "left": {"type": "Identifier", "name": "col1"}},
             "error": 'null'},
            {"alias": "new_col2", "expression": "len(col2, 3)",
                              "tree": {"type": "CallExpression",
                                       "callee": {"type": "Identifier",
                                                  "name": "len"}, "arguments": [
                                      {"type": "Identifier", "name": "col2"},
                                      {"raw": "3", "type": "Literal",
                                       "value": 3}]}, "error": 'null'},
            {"alias": "new_col3", "expression": "split(col3, ',')",
             "tree": {"type": "CallExpression",
                      "callee": {"type": "Identifier", "name": "split"},
                      "arguments": [{"type": "Identifier", "name": "col3"},
                                    {"raw": "','", "type": "Literal",
                                     "value": ","}]}, "error": 'null'}
        ]
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent(
        """
        {out} = {in1}.copy()
        
        functions = [['new_col1', lambda row: row['col1'] + 2 * 9],
                     ['new_col2', lambda row: len(row['col2'], 3)],
                     ['new_col3', lambda row: row['col3'].split(',')],]
        for col, function in functions:
            {out}[col] = {out}.apply(function, axis=1)
        """.format(out=n_out['output data'],
                   in1=n_in['input data'],
                   )
    )

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_math_expression_success():
    alias = 'result_2'
    expr = [{'tree': {
        "type": "BinaryExpression",
        "operator": "*",
        "left": {
            "type": "Identifier",
            "name": "a"
        },
        "right": {
            "type": "Literal",
            "value": 100,
            "raw": "100"
        }
    }, 'alias': alias, 'expression': "lower(a)"}]

    params = {
        TransformationOperation.EXPRESSION_PARAM: expr,
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent(
        """
        {out} = {in1}.copy()
        
        functions = [['result_2', lambda row: row['a'] * 100],]
        for col, function in functions:
            {out}[col] = {out}.apply(function, axis=1)
        """.format(out=n_out['output data'],
                   in1=n_in['input data']))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_missing_expr_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        TransformationOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


'''
    Union (Add-Rows) Operation
'''


def test_union_minimal_params_success():
    params = {}

    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = UnionOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = pd.concat([{in0}, {in1}], " \
                    "sort=False, axis=0, ignore_index=True)"\
        .format(out=n_out['output data'], in0=n_in['input data 1'],
                in1=n_in['input data 2'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_rows_minimal_params_success():
    params = {
        AggregationOperation.FUNCTION_PARAM: [
            {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
        AggregationOperation.ATTRIBUTES_PARAM: ['country']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = AggregationOperation(params, named_inputs=n_in,
                                    named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent("""
          def _collect_list(x):
              return x.tolist()
          
          def _merge_set(x):
              return set(x.tolist())
          
          
          columns = ['country']
          target = {'income': ['avg_income']}
          operations = {'income': ['AVG']}
          
          output_1 = input_1.groupby(columns).agg(operations)
          new_idx = []
          i = 0
          old = None
          for (n1, n2) in output_1.columns.ravel():
              if old != n1:
                  old = n1
                  i = 0
              new_idx.append(target[n1][i])
              i += 1
          
          output_1.columns = new_idx
          output_1 = output_1.reset_index()
          output_1.reset_index(drop=True, inplace=True)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


# def test_aggregation_rows_group_all_missing_attributes_success():
#     params = {
#         AggregationOperation.FUNCTION_PARAM: [
#             {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#
#     instance = AggregationOperation(params, named_inputs=n_in,
#                                     named_outputs=n_out)
#     code = instance.generate_code()
#
#     expected_code = """{out} = {in0}.agg(
#                         functions.avg('income').alias('avg_income'))""".format(
#         out=n_out['output data'], in0=n_in['input data'], agg='country', )
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     print (code)
#     assert result, msg


def test_aggregation_missing_function_param_failure():
    params = {
        AggregationOperation.ATTRIBUTES_PARAM: ['country']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    with pytest.raises(ValueError):
        AggregationOperation(params, named_inputs=n_in,
                             named_outputs=n_out)


def test_aggregation_with_pivot_success():
    params = {

        AggregationOperation.ATTRIBUTES_PARAM: ["sex"],
        AggregationOperation.FUNCTION_PARAM: [
            {"attribute": "fare", "f": "max", "alias": "sex"}],
        "pivot": ["class"],
        "pivot_values":  "",
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = AggregationOperation(params, named_inputs=n_in,
                                    named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent("""
    def _collect_list(x):
              return x.tolist()
    def _merge_set(x):
        return set(x.tolist())
    
    aggfunc = {'fare': ['max']}
    output_1 = pd.pivot_table(input_1, index=['sex'], values=['fare'],
                              columns=['class'], aggfunc=aggfunc)
    # rename columns and convert to DataFrame
    output_1.reset_index(inplace=True)
    new_idx = [n[0] if n[1] is ''
               else "%s_%s_%s" % (n[0],n[1], n[2])
               for n in output_1.columns.ravel()]    
    output_1 = pd.DataFrame(output_1.to_records())
    output_1.reset_index(drop=True, inplace=True)
    output_1 = output_1.drop(columns='index')
    output_1.columns = new_idx
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_with_pivot_values_success():
    params = {

        AggregationOperation.ATTRIBUTES_PARAM: ["sex"],
        AggregationOperation.FUNCTION_PARAM: [
            {"attribute": "fare", "f": "max", "alias": "sex"}],
        "pivot": ["class"],
        "pivot_values": [1, 2],
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = AggregationOperation(params, named_inputs=n_in,
                                    named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent("""
    def _collect_list(x):
              return x.tolist()
    def _merge_set(x):
        return set(x.tolist())

    values = [1, 2]
    input_1 = input_1[input_1['class'].isin(values)]
    aggfunc = {'fare': ['max']}
    output_1 = pd.pivot_table(input_1, index=['sex'], values=['fare'],
                              columns=['class'], aggfunc=aggfunc)
    # rename columns and convert to DataFrame
    output_1.reset_index(inplace=True)
    new_idx = [n[0] if n[1] is ''
               else "%s_%s_%s" % (n[0],n[1], n[2])
               for n in output_1.columns.ravel()]    
    output_1 = pd.DataFrame(output_1.to_records())
    output_1.reset_index(drop=True, inplace=True)
    output_1 = output_1.drop(columns='index')
    output_1.columns = new_idx
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg

