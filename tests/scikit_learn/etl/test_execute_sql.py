from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import ExecuteSQLOperation
import pytest
import pandasql
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# ExecuteSQL
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_execute_sql_success():
    df1 = util.iris(['class', 'sepalwidth'], size=10)
    test_df = df1.copy().sepalwidth > 3.2
    test_df = df1.copy()[test_df]
    test_df.reset_index(inplace=True, drop=True)

    arguments = {
        'parameters': {
            'query': 'SELECT class, sepalwidth FROM ds1 WHERE sepalwidth > 3.2',
            'names': 'class,sepalwidth'},
        'named_inputs': {
            'input data 1': 'df1'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df1': df1})
    assert result['out'].equals(test_df)


def test_execute_sql_names_param_not_informed_success():
    df1 = util.iris(['class', 'sepalwidth'], size=10)
    test_df = df1.copy()
    arguments = {
        'parameters': {'query': 'SELECT class, sepalwidth FROM ds1'},
        'named_inputs': {
            'input data 1': 'df1',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df1': df1})
    assert result['out'].equals(test_df)


def test_execute_sql_multiple_dataframes_success():
    df1 = util.iris(['class'], size=10)
    df2 = util.iris(['sepalwidth'], size=10)
    test_df = df1.copy().join(df2.copy())
    test_df.sort_values(by='sepalwidth', inplace=True)
    test_df.drop_duplicates(inplace=True, ignore_index=True)

    arguments = {
        'parameters': {'query': 'SELECT DISTINCT class, sepalwidth FROM ds1,'
                                ' ds2 ORDER BY class, sepalwidth',
                       'names': 'class,sepalwidth'},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df1': df1,
                           'df2': df2})
    assert result['out'].equals(test_df)


def test_execute_sql_no_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'query': 'SELECT class, sepalwidth FROM ds1 WHERE sepalwidth > 3.2',
            'names': 'class,sepalwidth'
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    assert instance.generate_code() is None


def test_execute_sql_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {
            'query': 'SELECT class, sepalwidth FROM ds1 WHERE sepalwidth > 3.2',
            'names': 'class,sepalwidth'
        },
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_execute_sql_column_not_found_fail():
    df1 = util.iris(['class', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'query': 'SELECT unknown FROM ds1'},
        'named_inputs': {
            'input data 1': 'df1',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    with pytest.raises(pandasql.PandaSQLException) as psql_err:
        util.execute(util.get_complete_code(instance),
                     {'df1': df1})
    assert "(sqlite3.OperationalError) no such column: unknown" in str(
        psql_err.value)


def test_execute_sql_wrong_number_of_attributes_informed_fail():
    df1 = util.iris(['class', 'sepalwidth'], size=10)

    arguments = {
        'parameters': {
            'query': 'SELECT class, sepalwidth FROM ds1',
            'names': 'class'},
        'named_inputs': {
            'input data 1': 'df1'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        util.execute(util.get_complete_code(instance),
                     {'df1': df1})
    assert "Invalid names. Number of attributes in" \
           " result differs from names informed." in str(val_err.value)


def test_execute_sql_missing_parameters_fail():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ExecuteSQLOperation(**arguments)
    assert "Required parameter query must be informed for task" in str(
        val_err.value)


def test_execute_sql_only_select_is_allowed_fail():
    arguments = {
        'parameters': {
            'query': 'UPDATE sepalwidth FROM ds1',
            'name': 'class,sepalwidth'},
        'named_inputs': {
            'input data 1': 'df1'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(ValueError) as val_err:
        ExecuteSQLOperation(**arguments)
    assert "Invalid query. Only SELECT is allowed." in str(val_err.value)
