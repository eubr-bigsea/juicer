from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import JoinOperation
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Join
#


def test_join_success():
    df1 = ['df1', util.titanic(['name', 'homedest'], 2)]

    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    df1_tst = util.titanic(['name', 'homedest'], 2)
    df2_tst = util.titanic(['embarked', 'name'], 10)

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'inner',
                       'left_attributes': ['name'],
                       'right_attributes': ['name']},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    cols1 = [c + '_l' for c in df1_tst.columns]
    cols2 = [c + '_r' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_l' for c in ['name']]
    keys2 = [c + '_r' for c in ['name']]

    out = pd.merge(df1_tst, df2_tst, how='inner',
                   suffixes=['_l', '_r'],
                   left_on=keys1, right_on=keys2)

    out.drop(columns=['name_r'], inplace=True)

    assert result['out'].equals(out)


def test_join_krk_param_success():
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]
    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    df1_tst = util.titanic(['name', 'homedest'], 10)
    df2_tst = util.titanic(['embarked', 'name'], 10)

    arguments = {
        'parameters': {
            'keep_right_keys': 1, 'match_case': '1',
            'join_type': 'inner', 'left_attributes': ['name'],
            'right_attributes': ['name']
        },
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df1': df1[1],
                                                     'df2': df2[1]})

    cols1 = [c + '_l' for c in df1_tst.columns]
    cols2 = [c + '_r' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_l' for c in ['name']]
    keys2 = [c + '_r' for c in ['name']]

    out = pd.merge(df1_tst, df2_tst, how='inner',
                   suffixes=['_l', '_r'],
                   left_on=keys1, right_on=keys2)

    assert result['out'].equals(out)


def test_join_match_case_param_success():
    # Match case converts a column to lower
    # then, it adds a _lower to the column name
    # and finally it drops the column
    # redundant?

    df1 = ['df1', util.titanic(['name', 'embarked'], 10)]
    df2 = ['df2', util.titanic(['homedest', 'name'], 10)]

    df1_tst = util.titanic(['name', 'embarked'], 10)
    df2_tst = util.titanic(['homedest', 'name'], 10)

    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': ['name'],
            'right_attributes': ['name']
        },
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df1': df1[1],
                                                     'df2': df2[1]})

    cols1 = [c + '_l' for c in df1_tst.columns]
    cols2 = [c + '_r' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_l' for c in ['name']]
    keys2 = [c + '_r' for c in ['name']]

    data1_tmp = df1_tst[keys1].applymap(lambda col: str(col).lower()).copy()
    data1_tmp.columns = [c + "_lower" for c in data1_tmp.columns]
    col1 = list(data1_tmp.columns)
    data1_tmp = pd.concat([df1_tst, data1_tmp], axis=1, sort=False)

    data2_tmp = df2_tst[keys2].applymap(lambda col: str(col).lower()).copy()
    data2_tmp.columns = [c + "_lower" for c in data2_tmp.columns]
    col2 = list(data2_tmp.columns)
    data2_tmp = pd.concat([df2_tst, data2_tmp], axis=1, sort=False)

    out = pd.merge(data1_tmp, data2_tmp, left_on=col1, right_on=col2,
                   copy=False, suffixes=['_l', '_r'], how='inner')
    # Why drop col_lower?
    # out.drop(col1+col2, axis=1, inplace=True)

    cols_to_remove = keys2
    out.drop(cols_to_remove, axis=1, inplace=True)

    assert result['out'].equals(out)


def test_join_custom_suffixes_success():
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]
    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    df1_tst = util.titanic(['name', 'homedest'], 10)
    df2_tst = util.titanic(['embarked', 'name'], 10)

    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': True,
            'join_type': 'inner', 'left_attributes': ['name'],
            'right_attributes': ['name'], 'aliases': '_esquerdo,_direito'
        },
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df1': df1[1],
                                                     'df2': df2[1]})

    cols1 = [c + '_esquerdo' for c in df1_tst.columns]
    cols2 = [c + '_direito' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_esquerdo' for c in ['name']]
    keys2 = [c + '_direito' for c in ['name']]

    out = pd.merge(df1_tst, df2_tst, how='inner',
                   suffixes=['_esquerdo', '_direito'],
                   left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    out.drop(cols_to_remove, axis=1, inplace=True)

    assert result['out'].equals(out)


def test_join_merge_outer_parameter_success():
    # there's a line of code that replaces '_outer' to ''
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]

    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    df1_tst = util.titanic(['name', 'homedest'], 10)
    df2_tst = util.titanic(['embarked', 'name'], 10)

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'outer',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    cols1 = [c + '_l' for c in df1_tst.columns]
    cols2 = [c + '_r' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_l' for c in ['homedest']]
    keys2 = [c + '_r' for c in ['embarked']]

    out = pd.merge(df1_tst, df2_tst, how='outer',
                   suffixes=['_l', '_r'],
                   left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    out.drop(cols_to_remove, axis=1, inplace=True)
    assert result['out'].equals(out)


def test_join_merge_left_parameter_success():
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]

    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    df1_tst = util.titanic(['name', 'homedest'], 10)
    df2_tst = util.titanic(['embarked', 'name'], 10)

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'left',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    cols1 = [c + '_l' for c in df1_tst.columns]
    cols2 = [c + '_r' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_l' for c in ['homedest']]
    keys2 = [c + '_r' for c in ['embarked']]

    out = pd.merge(df1_tst, df2_tst, how='left',
                   suffixes=['_l', '_r'],
                   left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    out.drop(cols_to_remove, axis=1, inplace=True)
    assert result['out'].equals(out)


def test_join_merge_right_parameter_success():
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]

    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    df1_tst = util.titanic(['name', 'homedest'], 10)
    df2_tst = util.titanic(['embarked', 'name'], 10)

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'right',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    cols1 = [c + '_l' for c in df1_tst.columns]
    cols2 = [c + '_r' for c in df2_tst.columns]

    df1_tst.columns = cols1
    df2_tst.columns = cols2

    keys1 = [c + '_l' for c in ['homedest']]
    keys2 = [c + '_r' for c in ['embarked']]

    out = pd.merge(df1_tst, df2_tst, how='right',
                   suffixes=['_l', '_r'],
                   left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    out.drop(cols_to_remove, axis=1, inplace=True)
    assert result['out'].equals(out)


def test_join_outer_replace_success():
    # This only happens when you pass '_outer'
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]
    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': '_outer',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        instance = JoinOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df1': df1[1],
                               'df2': df2[1]})
    assert '' in str(key_err)


def test_join_missing_attributes_success():
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]
    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner'
        },
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = JoinOperation(**arguments)
        result = util.execute(instance.generate_code(), {'df1': df1[1],
                                                         'df2': df2[1]})
    assert "Parameters 'left_attributes' and 'right_attributes'" \
           " must be informed for task" in str(val_err)


def test_join_success_no_output_value_error():
    df1 = ['df1', util.titanic(['name', 'homedest'], 10)]
    df2 = ['df2', util.titanic(['embarked', 'name'], 10)]

    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': ['homedest'],
            'right_attributes': ['embarked']
        },
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {}
    }
    with pytest.raises(ValueError) as val_err:
        instance = JoinOperation(**arguments)
        result = util.execute(instance.generate_code(), {'df1': df1[1],
                                                         'df2': df2[1]})
    assert "Parameter 'input data 1', 'input data 2' and " \
           "'named_outputs' must be informed for task" in str(val_err)