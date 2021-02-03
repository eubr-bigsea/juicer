from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import JoinOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Join
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_join_success():
    df1 = util.titanic(['name', 'homedest'], size=2)
    df2 = util.titanic(['embarked', 'name'], size=10)
    test_df = util.titanic(['name', 'homedest', 'embarked'], size=2)

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'inner',
                       'left_attributes': ['name'],
                       'right_attributes': ['name']},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    test_df.columns = ['name_l', 'homedest_l', 'embarked_r']
    assert result['out'].equals(test_df)


def test_join_krk_param_success():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    test_df = util.titanic(['name', 'homedest', 'embarked', 'name'], size=10)

    arguments = {
        'parameters': {
            'keep_right_keys': 1, 'match_case': '1',
            'join_type': 'inner', 'left_attributes': ['name'],
            'right_attributes': ['name']
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df1': df1,
                                                     'df2': df2})
    test_df.columns = ['name_l', 'homedest_l', 'embarked_r', 'name_r']
    assert result['out'].equals(test_df)


def test_join_match_case_param_success():
    """
    Match case converts a column to lower then, it adds a _lower to the column
    name and finally it drops the column. (Seems redundant...)
    """
    df1 = util.titanic(['name', 'embarked'], size=10)
    df2 = util.titanic(['homedest', 'name'], size=10)
    test_df = util.titanic(['name', 'embarked', 'homedest'], size=10)

    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': ['name'],
            'right_attributes': ['name']
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df1': df1,
                                                     'df2': df2})

    test_df.columns = ['name_l', 'embarked_l', 'homedest_r']
    assert result['out'].equals(test_df)


def test_join_custom_suffixes_success():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    test_df = util.titanic(['name', 'homedest', 'embarked'], size=10)

    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': True,
            'join_type': 'inner', 'left_attributes': ['name'],
            'right_attributes': ['name'], 'aliases': '_esquerdo,_direito'
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df1': df1,
                                                     'df2': df2})
    test_df.columns = ['name_esquerdo', 'homedest_esquerdo', 'embarked_direito']
    assert result['out'].equals(test_df)


def test_join_merge_outer_parameter_success():
    """
    there's a line of code that replaces '_outer' to ''
    """
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    test_df1 = df1.copy()
    test_df2 = df2.copy()

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'outer',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    cols1 = [c + '_l' for c in test_df1.columns]
    cols2 = [c + '_r' for c in test_df2.columns]

    test_df1.columns = cols1
    test_df2.columns = cols2

    keys1 = [c + '_l' for c in ['homedest']]
    keys2 = [c + '_r' for c in ['embarked']]

    test_out = pd.merge(test_df1, test_df2, how='outer',
                        suffixes=['_l', '_r'],
                        left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    test_out.drop(cols_to_remove, axis=1, inplace=True)
    assert result['out'].equals(test_out)


def test_join_merge_left_parameter_success():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    test_df1 = df1.copy()
    test_df2 = df2.copy()

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'left',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    cols1 = [c + '_l' for c in test_df1.columns]
    cols2 = [c + '_r' for c in test_df2.columns]

    test_df1.columns = cols1
    test_df2.columns = cols2

    keys1 = [c + '_l' for c in ['homedest']]
    keys2 = [c + '_r' for c in ['embarked']]

    test_out = pd.merge(test_df1, test_df2, how='left',
                        suffixes=['_l', '_r'],
                        left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    test_out.drop(cols_to_remove, axis=1, inplace=True)
    assert result['out'].equals(test_out)


def test_join_merge_right_parameter_success():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    test_df1 = df1.copy()
    test_df2 = df2.copy()

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'right',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    cols1 = [c + '_l' for c in test_df1.columns]
    cols2 = [c + '_r' for c in test_df2.columns]

    test_df1.columns = cols1
    test_df2.columns = cols2

    keys1 = [c + '_l' for c in ['homedest']]
    keys2 = [c + '_r' for c in ['embarked']]

    test_out = pd.merge(test_df1, test_df2, how='right',
                        suffixes=['_l', '_r'],
                        left_on=keys1, right_on=keys2)

    cols_to_remove = keys2
    test_out.drop(cols_to_remove, axis=1, inplace=True)
    assert result['out'].equals(test_out)


def test_join_outer_replace_success():
    """
    This only happens when you pass '_outer'
    """
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)

    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': '_outer',
                       'left_attributes': ['homedest'],
                       'right_attributes': ['embarked']},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df1': df1,
                      'df2': df2})
    assert '' in str(key_err.value)


def test_join_no_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': ['homedest'],
            'right_attributes': ['embarked']
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {}
    }
    instance = JoinOperation(**arguments)
    assert instance.generate_code() is None


def test_join_missing_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': ['homedest'],
            'right_attributes': ['embarked']
        },
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_join_missing_attributes_param_fail():
    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner'
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(ValueError) as val_err:
        JoinOperation(**arguments)
    assert "Parameters 'left_attributes' and 'right_attributes'" \
           " must be informed for task" in str(val_err.value)


def test_join_invalid_join_type_param_fail():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'invalid', 'left_attributes': ['homedest'],
            'right_attributes': ['embarked']
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df1': df1,
                      'df2': df2})
    assert "invalid" in str(key_err.value)


def test_join_invalid_left_attributes_param_fail():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': 'invalid',
            'right_attributes': ['embarked']
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(instance.generate_code(),
                     {'df1': df1,
                      'df2': df2})
    assert "invalid" in str(nam_err.value)


def test_join_invalid_right_attributes_param_fail():
    df1 = util.titanic(['name', 'homedest'], size=10)
    df2 = util.titanic(['embarked', 'name'], size=10)
    arguments = {
        'parameters': {
            'keep_right_keys': False, 'match_case': False,
            'join_type': 'inner', 'left_attributes': ['homedest'],
            'right_attributes': 'invalid'
        },
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = JoinOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(instance.generate_code(),
                     {'df1': df1,
                      'df2': df2})
    assert "invalid" in str(nam_err.value)
