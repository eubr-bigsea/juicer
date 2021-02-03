from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import IntersectionOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Intersection
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_intersection_success():
    df1 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=10)
    df2 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=20)

    expectedkeys = df1.columns.tolist()
    expectedresult = pd.merge(df1, df2, how='inner', on=expectedkeys,
                              indicator=False, copy=False)
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
    instance = IntersectionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1, 'df2': df2})
    assert result['out'].equals(expectedresult)


def test_intersection_success_missing_input_inplies_no_code():
    # deve dar sucessso porem sem gerar codigo, has_code() deve retornar False
    arguments = {
        'parameters': {},
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        IntersectionOperation(**arguments)
    assert "Parameter 'input data 1' and 'input data 2' must be informed" \
           " for task" in str(exc_info.value)


def test_intersection_success_missing_output_inplies_no_code():
    # deve dar sucessso porem sem gerar codigo
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
        }
    }
    with pytest.raises(ValueError) as exc_info:
        IntersectionOperation(**arguments)
    assert "Parameter 'input data 1' and 'input data 2' must be informed" \
           " for task" in str(exc_info.value)


# # # # # # # # # # Fail # # # # # # # # # #
def test_1_input():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        IntersectionOperation(**arguments)
    assert "Parameter 'input data 1' and 'input data 2' must be informed" \
           " for task" in str(exc_info.value)


def test_3_input():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2',
            'input data 3': 'df3'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        IntersectionOperation(**arguments)
    assert "Parameter 'input data 1' and 'input data 2' must be informed" \
           " for task" in str(exc_info.value)


def test_intersection_fail_different_colunms():
    df1 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=10)
    df2 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'class'], size=20)
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
    instance = IntersectionOperation(**arguments)
    with pytest.raises(KeyError) as exc_info:
        util.execute(instance.generate_code(),
                     {'df1': df1, 'df2': df2})
    assert 'petalwidth' in str(exc_info.value)


def test_intersection_fail_different_number_of_colunms():
    df1 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=10)
    df2 = util.iris(['sepallength', 'sepalwidth',
                     'petallength'], size=20)
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
    instance = IntersectionOperation(**arguments)
    with pytest.raises(ValueError) as exc_info:
        util.execute(instance.generate_code(),
                     {'df1': df1, 'df2': df2})
    assert 'For intersection operation, both input data sources must have the' \
           ' same number of attributes and types.' in str(exc_info.value)


def test_interssection_fail_input_data_1_empty():
    df1 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=0)
    df2 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=20)

    expectedkeys = df1.columns.tolist()
    expectedresult = pd.merge(df1, df2, how='inner', on=expectedkeys,
                              indicator=False, copy=False)
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
    instance = IntersectionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1, 'df2': df2})
    assert result['out'].equals(expectedresult)


def test_interssection_fail_input_data_2_empty():
    df1 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=10)
    df2 = util.iris(['sepallength', 'sepalwidth',
                     'petallength', 'petalwidth'], size=0)

    expectedkeys = df1.columns.tolist()
    expectedresult = pd.merge(df1, df2, how='inner', on=expectedkeys,
                              indicator=False, copy=False)
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
    instance = IntersectionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1, 'df2': df2})
    assert result['out'].equals(expectedresult)
