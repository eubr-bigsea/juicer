from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AddColumnsOperation
import pytest
from juicer.operation import Operation
from ..fixtures import get_parametrize

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Add columns operation
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(**get_parametrize('etl', 'AddColumnsOperation'))
@pytest.mark.parametrize('size', [10, 5])
@pytest.mark.parametrize('aliases', [None, '_value0,_value1'])
def test_add_columns_success(impl: Operation, source: callable, target: callable,
                             size: int, aliases: str, request):
    left_df = util.iris(['sepallength', 'sepalwidth', 'class'], size=10)
    right_df = util.iris(['petallength', 'petalwidth', 'class'], size=size)

    con = None
    if request.node.callspec.id == 'duckdb':
        pass
        # con = duckdb.connect()
        # import builtins
        # builtins.__dict__['get_global_duckdb_conn'] = lambda: con
        # l_df = con.query('SELECT * FROM left_df')
        # r_df = con.query('SELECT * FROM right_df')
    else:
        l_df = source(left_df)
        r_df = source(right_df)

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    if aliases:
        arguments['parameters']['aliases'] = aliases

    instance = impl(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df1': l_df, 'df2': r_df})

    result_df = target(result['out'])
    test_df = util.iris([
        'sepallength', 'sepalwidth',
        'petallength', 'petalwidth', 'class'], size=10)
    test_df.insert(2, 'class_ds0', test_df['class'])

    if aliases:
        test_df.columns = [
            'sepallength', 'sepalwidth', 'class_value0',
            'petallength', 'petalwidth', 'class_value1']
    else:
        test_df.columns = [
            'sepallength', 'sepalwidth', 'class_ds0',
            'petallength', 'petalwidth', 'class_ds1']

    if size != 10:
        # Simulate the merge when the number of rows is different
        test_df.iloc[5:, 3:] = None

    assert result_df.equals(test_df)


def test_add_columns_repeated_column_names_success():
    left_df = util.iris(['sepallength', 'class'], size=10)
    right_df = util.iris(['sepallength', 'class'], size=10)
    test_df = util.iris(
        ['sepallength', 'class', 'sepallength', 'class'], size=10)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df, 'df2': right_df})
    test_df.columns = ['sepallength_ds0', 'class_ds0',
                       'sepallength_ds1', 'class_ds1']
    assert result['out'].equals(test_df)


def test_add_columns_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
        }
    }
    instance = AddColumnsOperation(**arguments)
    assert instance.generate_code() is None


def test_add_columns_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_add_columns_invalid_aliases_param_fail():
    left_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    right_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'aliases': 'invalid'},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    with pytest.raises(IndexError) as idx_err:
        util.execute(instance.generate_code(),
                     {'df1': left_df, 'df2': right_df})
    assert 'list index out of range' in str(idx_err.value)
