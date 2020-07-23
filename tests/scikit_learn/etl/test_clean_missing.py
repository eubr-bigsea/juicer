from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import CleanMissingOperation
import pandas as pd
import numpy as np
import pytest


# CleanMissing
#
def test_clean_missing_fail_multiplicity_parameter_not_passed():
    # Multiplicity adds '.copy()' to the resulting code, but it works
    # in a weird way. Is it intended to work like this?
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEAN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})


def test_clean_missing_success_missing_parameters():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = CleanMissingOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})

    assert ('Parameter \'attributes\' must be informed for task' in str(val_err))


def test_clean_missing_success_missing_attributes_parameter():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'cleaning_mode': 'MEDIAN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = CleanMissingOperation(**arguments)
        result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert('Parameter \'attributes\' must be informed for task' in str(val_err))


def test_clean_missing_success_missing_cleaning_mode_parameter():
    # Defaults to REMOVE_ROW
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert result['output_data_1'].equals(util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], 10).drop(index=0))


def test_clean_missing_success_fill_value_missing_value_parameter():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'VALUE'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = CleanMissingOperation(**arguments)
        result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert('Parameter \'value\' must be not None when mode is \'VALUE\' for task'
    in str(val_err))


def test_clean_missing_success_missing_named_inputs():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEAN'},
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    with pytest.raises(AttributeError) as att_err:
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert ('\'CleanMissingOperation\' object has no attribute \'mode_CM\'' in
            str(att_err))


def test_clean_missing_success_no_output_implies_no_code():
    # It's generating code without output
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEAN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {}
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert not instance.has_code


def test_clean_missing_fill_value_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'VALUE',
                       'value': 'replaced'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].loc[0, 'sepalwidth'] == 'replaced'


def test_clean_missing_fill_median_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN
    sepal_median = df[1].loc[:, 'sepalwidth']

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEDIAN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].loc[0, 'sepalwidth'] == sepal_median.median()


def test_clean_missing_fill_mode_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MODE'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    sepal_mode = util.iris(['sepalwidth'], 10).loc[:, 'sepalwidth'].mode()
    assert result['output_data_1'].loc[0, 'sepalwidth'] == sepal_mode[0]


def test_clean_missing_fill_mean_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, 'sepalwidth'] = np.NaN
    sepal_mean = df[1].loc[:, 'sepalwidth']

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'MEAN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].loc[0, 'sepalwidth'] == sepal_mean.mean()


def test_clean_missing_remove_row_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[4, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'REMOVE_ROW'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].equals(
        util.iris(['sepallength', 'sepalwidth',
                   'petalwidth', 'petallength'], 10).drop(index=4))


def test_clean_missing_remove_column_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[4, 'sepalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepalwidth'],
                       'cleaning_mode': 'REMOVE_COLUMN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].equals(
        util.iris(['sepallength', 'sepalwidth', 'petalwidth', 'petallength'],
                  10).drop(columns=['sepalwidth']))


def test_clean_missing_multiple_attributes_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0:1, 'sepallength'] = np.NaN
    df[1].loc[2:3, 'sepalwidth'] = np.NaN
    df[1].loc[4:5, 'petalwidth'] = np.NaN
    df[1].loc[6:7, 'petallength'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepallength', 'sepalwidth',
                                      'petalwidth', 'petallength'],
                       'cleaning_mode': 'REMOVE_ROW'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].equals(
        util.iris(['sepallength', 'sepalwidth',
                   'petalwidth', 'petallength'], 10).drop(
            index=[i for i in range(8)]))


def test_clean_missing_ratio_control_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[:, ['sepallength', 'sepalwidth']] = np.NaN
    df[1].loc[0, 'petalwidth'] = np.NaN

    test = util.iris(['sepallength', 'sepalwidth',
                      'petalwidth', 'petallength'], 10)
    test.loc[:, ['sepallength', 'sepalwidth']] = np.NaN
    test.loc[0, 'petalwidth'] = np.NaN

    # ratio = df[col].isnull().sum() / len(df)
    # ratio_sepallength = 10/40 (0.25)
    # ratio_sepalwidth = 10/40 (0.25)
    # ratio_petalwidth = 1/40 (0.025)

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepallength', 'sepalwidth',
                                      'petalwidth', 'petallength'],
                       'min_missing_ratio': 0.025,
                       'max_missing_ratio': 0.1,
                       'cleaning_mode': 'REMOVE_COLUMN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CleanMissingOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['output_data_1'].equals(
        test.drop(columns=['petalwidth']))


def test_clean_missing_max_ratio_is_lower_than_min_ratio_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[:, ['sepallength', 'sepalwidth']] = np.NaN
    df[1].loc[0, 'petalwidth'] = np.NaN

    test = util.iris(['sepallength', 'sepalwidth',
                      'petalwidth', 'petallength'], 10)
    test.loc[:, ['sepallength', 'sepalwidth']] = np.NaN
    test.loc[0, 'petalwidth'] = np.NaN

    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'attributes': ['sepallength', 'sepalwidth',
                                      'petalwidth', 'petallength'],
                       'min_missing_ratio': 0.25,
                       'max_missing_ratio': 0.025,
                       'cleaning_mode': 'REMOVE_COLUMN'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = CleanMissingOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert ('Parameter \'attributes\' must be 0<=x<=1 for task' in str(val_err))

# TODO

# Fix multiplicity(?)