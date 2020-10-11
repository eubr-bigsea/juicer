from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import ExecuteSQLOperation
import pytest
import pandasql
import pandas as pd

pd.set_option('display.max_rows', None)


# ExecuteSQL
# 
def test_execute_sql_success():
    slice_size = 10
    df1 = ['df1', util.iris(['class', 'sepalwidth'], slice_size)]

    arguments = {
        'parameters': {
            'query': 'SELECT class, sepalwidth FROM ds1 WHERE sepalwidth > 3.2',
            'names': 'class,sepalwidth'},
        'named_inputs': {
            'input data 1': df1[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1]})
    tst = df1[1].sepalwidth > 3.2
    tst = df1[1][tst]
    tst.reset_index(inplace=True, drop=True)
    assert result['out'].equals(tst)


def test_execute_sql_success_names_not_informed():
    slice_size = 10
    df1 = ['df1', util.iris(['class', 'sepalwidth'], slice_size)]
    arguments = {
        'parameters': {'query': 'SELECT class, sepalwidth FROM ds1'},
        'named_inputs': {
            'input data 1': df1[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1]})
    assert result['out'].equals(df1[1])


def test_execute_sql_success_multiple_frames():
    slice_size = 10
    df1 = ['df1', util.iris(['class'], slice_size)]
    df2 = ['df2', util.iris(['sepalwidth'], slice_size)]

    arguments = {
        'parameters': {'query': 'SELECT DISTINCT class, sepalwidth FROM ds1,'
                                ' ds2 ORDER BY class, sepalwidth',
                       'names': 'class,sepalwidth'},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ExecuteSQLOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})
    tst = df1[1].join(df2[1])
    tst.sort_values(by='sepalwidth', inplace=True)
    tst.drop_duplicates(inplace=True, ignore_index=True)
    assert result['out'].equals(tst)


def test_execute_sql_success_column_not_found():
    slice_size = 10
    df1 = ['df1', util.iris(['class', 'sepalwidth'], slice_size)]
    arguments = {
        'parameters': {'query': 'SELECT unknown FROM ds1'},
        'named_inputs': {
            'input data 1': df1[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(pandasql.PandaSQLException) as psql_err:
        instance = ExecuteSQLOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df1': df1[1]})
    assert "(sqlite3.OperationalError) no such column: unknown" in str(psql_err)


def test_execute_sql_success_wrong_number_of_attributes_informed():
    slice_size = 10
    df1 = ['df1', util.iris(['class', 'sepalwidth'], slice_size)]

    arguments = {
        'parameters': {
            'query': 'SELECT class, sepalwidth FROM ds1',
            'names': 'class'},
        'named_inputs': {
            'input data 1': df1[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = ExecuteSQLOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df1': df1[1]})
    assert "Invalid names. Number of attributes in" \
           " result differs from names informed." in str(val_err)


def test_execute_sql_success_only_select_is_allowed():
    slice_size = 10
    df1 = ['df1', util.iris(['class', 'sepalwidth'], slice_size)]

    arguments = {
        'parameters': {
            'query': 'UPDATE sepalwidth FROM ds1',
            'name': 'class,sepalwidth'},
        'named_inputs': {
            'input data 1': df1[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = ExecuteSQLOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df1': df1[1]})
    assert "Invalid query. Only SELECT is allowed." in str(val_err)
