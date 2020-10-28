from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import CleanMissingOperation
import numpy as np
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# CleanMissing
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_clean_missing_fill_value_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[0, 'sepalwidth'] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'VALUE',
                       'value': 'replaced'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].loc[0, 'sepalwidth'] == 'replaced'


def test_clean_missing_fill_median_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[0, 'sepalwidth'] = np.NaN
    sepal_median = df.copy().loc[:, 'sepalwidth']
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEDIAN'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].loc[0, 'sepalwidth'] == sepal_median.median()


def test_clean_missing_fill_mode_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[0, 'sepalwidth'] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MODE'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    sepal_mode = util.iris(['sepalwidth'], size=10).loc[:, 'sepalwidth'].mode()
    assert result['output_data_1'].loc[0, 'sepalwidth'] == sepal_mode[0]


def test_clean_missing_fill_mean_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[0, 'sepalwidth'] = np.NaN
    sepal_mean = df.copy().loc[:, 'sepalwidth']
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEAN'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].loc[0, 'sepalwidth'] == sepal_mean.mean()


def test_clean_missing_remove_row_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[4, 'sepalwidth'] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'REMOVE_ROW'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].equals(
        util.iris(['sepallength', 'sepalwidth',
                   'petalwidth', 'petallength'], size=10).drop(index=4))


def test_clean_missing_remove_column_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[4, 'sepalwidth'] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'REMOVE_COLUMN'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].equals(
        util.iris(['sepallength', 'sepalwidth', 'petalwidth', 'petallength'],
                  10).drop(columns=['sepalwidth']))


def test_clean_missing_multiple_attributes_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[0:1, 'sepallength'] = np.NaN
    df.loc[2:3, 'sepalwidth'] = np.NaN
    df.loc[4:5, 'petalwidth'] = np.NaN
    df.loc[6:7, 'petallength'] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth',
                                      'petalwidth', 'petallength'],
                       'cleaning_mode': 'REMOVE_ROW'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].equals(
        util.iris(['sepallength', 'sepalwidth',
                   'petalwidth', 'petallength'], size=10).drop(
            index=[i for i in range(8)]))


def test_clean_missing_missing_cleaning_mode_param_success():
    """
    Defaults to REMOVE_ROW
    """
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth',
                                      'petalwidth', 'petallength']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    util.execute(instance.generate_code(),
                 {'df': df})


def test_clean_missing_ratio_control_success():
    """
    Needs a better assertion...
    Ratio method is confusing.
    """
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[:, ['sepallength', 'sepalwidth']] = np.NaN
    df.loc[0, 'petalwidth'] = np.NaN
    test = util.iris(['sepallength', 'sepalwidth',
                      'petalwidth', 'petallength'], size=10)
    test.loc[:, ['sepallength', 'sepalwidth']] = np.NaN
    test.loc[0, 'petalwidth'] = np.NaN

    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth',
                                      'petalwidth', 'petallength'],
                       'min_missing_ratio': 0.025,
                       'max_missing_ratio': 0.1,
                       'cleaning_mode': 'REMOVE_COLUMN'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['output_data_1'].equals(
        test.drop(columns=['petalwidth']))


def test_clean_missing_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'VALUE',
                       'value': 'replaced'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = CleanMissingOperation(**arguments)
    assert instance.generate_code() is None


def test_clean_missing_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'VALUE',
                       'value': 'replaced'},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_clean_missing_missing_attributes_param_fail():
    arguments = {
        'parameters': {'cleaning_mode': 'REMOVE_ROW'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        CleanMissingOperation(**arguments)
    assert "'attributes' must be informed for task" in str(
        val_err.value)


def test_clean_missing_fill_value_missing_value_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.loc[0, 'sepalwidth'] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
                       'cleaning_mode': 'VALUE'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(ValueError) as val_err:
        CleanMissingOperation(**arguments)
    assert "Parameter 'value' must be not None when mode is 'VALUE' for task" \
           in str(val_err.value)


def test_clean_missing_max_ratio_is_lower_than_min_ratio_fail():
    arguments = {
        'parameters': {
            'attributes': ['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'],
            'min_missing_ratio': 0.25,
            'max_missing_ratio': 0.025,
            'cleaning_mode': 'REMOVE_COLUMN'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        CleanMissingOperation(**arguments)
    assert "Parameter 'attributes' must be 0<=x<=1 for task" in str(
        val_err.value)
