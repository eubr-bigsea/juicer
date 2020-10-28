from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SampleOrPartitionOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# SampleOrPartition
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_sample_or_partition_percent_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {
            'type': 'percent', 'fraction': 60
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    assert len(result['out']) == 6
    assert not result['out'].equals(test_df.iloc[:6, :])


def test_sample_or_partition_head_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'type': 'head', 'value': 2
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert len(result['out']) == 2
    assert result['out'].equals(test_df.iloc[:2, :])


def test_sample_or_partition_seed_success():
    """
    seeds 4294967296 or higher (integer limit) will be set to 0
    seeds lower than 0 will be set to 0
    """
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {
            'type': 'value', 'seed': 4294967296, 'value': 10
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df.sample(n=10, random_state=0)

    assert result['out'].equals(test_out)


def test_sample_or_partition_no_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'type': 'percent', 'fraction': 60
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    assert instance.generate_code() is None


def test_sample_or_partition_missing_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'type': 'percent', 'fraction': 60
        },
        'named_inputs': {
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_sample_or_partition_invalid_fraction_param_fail():
    arguments = {
        'parameters': {
            'type': 'percent', 'fraction': 110
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SampleOrPartitionOperation(**arguments)
    assert "Parameter 'fraction' must be 0<=x<=1 if is using the current type" \
           " of sampling in task" in str(val_err.value)


def test_sample_or_partition_invalid_value_param_fail():
    arguments = {
        'parameters': {
            'type': 'head', 'value': -1
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SampleOrPartitionOperation(**arguments)
    assert "Parameter 'value' must be [x>=0] if is using the current type of" \
           " sampling in task" in str(val_err.value)


def test_sample_or_partition_missing_parameters_fail():
    arguments = {
        'parameters': {
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SampleOrPartitionOperation(**arguments)
    assert "Parameter 'fraction' must be 0<=x<=1 if is using the current type of" \
           " sampling in task" in str(val_err)
