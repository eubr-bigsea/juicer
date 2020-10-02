from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SampleOrPartitionOperation
import pytest


# SampleOrPartition
# 
def test_sample_or_partition_percent_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            'type': 'percent', 'fraction': 60
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert len(result['out']) == 6
    assert not result['out'].equals(df[1].loc[:5, :])


def test_sample_or_partition_head_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]
    arguments = {
        'parameters': {
            'type': 'head', 'value': 2
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert len(result['out']) == 2


def test_sample_or_partition_seed_success():
    # seed 4294967296 or higher (integer limit) will be set to 0
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df_tst = util.iris(['sepallength', 'sepalwidth',
                        'petalwidth', 'petallength'], slice_size)

    arguments = {
        'parameters': {
            'type': 'anything', 'seed': 4294967296, 'value': 10
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    instance = SampleOrPartitionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    out = df_tst.sample(n=10, random_state=0)

    assert result['out'].equals(out)


def test_sample_or_partition_percent_bad_fraction_param_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            'type': 'percent', 'fraction': 110
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = SampleOrPartitionOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert "Parameter 'fraction' must be 0<=x<=1 if is using the current type" \
           " of sampling in task" in str(val_err)


def test_sample_or_partition_bad_value_param_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]
    arguments = {
        'parameters': {
            'type': 'head', 'value': -1
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'sampled data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = SampleOrPartitionOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert "Parameter 'value' must be [x>=0] if is using the current type of" \
           " sampling in task" in str(val_err)