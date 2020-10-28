from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import UnionOperation
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Union
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_union_success():
    df1 = util.iris(['sepallength', 'sepalwidth'], size=10)
    df2 = util.iris(['petalwidth', 'petallength'], size=10)
    test_df1 = df1.copy()
    test_df2 = df2.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = UnionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1, 'df2': df2})

    assert result['out'].equals(
        pd.concat([test_df1, test_df2], sort=False, axis=0, ignore_index=True))
    assert len(result['out']) == 20


def test_union_uneven_dataframe_sizes_success():
    df1 = util.iris(['sepallength', 'sepalwidth', ], size=5)
    df2 = util.iris(['petalwidth', 'petallength'], size=10)
    test_df1 = df1.copy()
    test_df2 = df2.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = UnionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1, 'df2': df2})

    assert len(result['out']) == 15
    test_out = pd.concat([test_df1, test_df2], sort=False, axis=0,
                         ignore_index=True)
    assert result['out'].equals(test_out)


def test_union_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'inpút data 2': 'df2'
        },
        'named_outputs': {}
    }

    instance = UnionOperation(**arguments)
    assert instance.generate_code() is None


def test_union_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    instance = UnionOperation(**arguments)
    assert instance.generate_code() is None
