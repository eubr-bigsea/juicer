from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SplitKFoldOperation
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def percent(df, n_groups):
    alpha = {}
    for i in range(n_groups):
        beta = df.loc[df['groups'] == i]
        beta = beta.groupby('class').size().apply(lambda x: 100 * x / len(beta))
        alpha.update({f"group{i}": beta})
    return alpha


# SplitKFold
#
def test_perfect_split_k_fold_success():
    # One class per group
    df = ['df', util.iris(['class'], 30)]

    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:30, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(df=result['out'], n_groups=3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == [100, 100, 100]
    assert result['out'].equals(tst)


def test_perfect_split_k_fold_shuffle_success():
    # Balanced percentage of each class in each group
    df = ['df', util.iris(['class'], 30)]

    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:30, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 1, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 18},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    data = {
        "groups": [
            0, 0, 2, 0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 0, 2, 1, 1, 2, 2, 2, 0, 0, 2,
            1, 2, 1, 1, 1, 0, 1
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(df=result['out'], n_groups=3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group1']['Iris-setosa'],
            count_percent['group2']['Iris-setosa']] == [50, 20, 30]

    assert [count_percent['group0']['Iris-versicolor'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group2']['Iris-versicolor']] == [20, 30, 50]

    assert [count_percent['group0']['Iris-virginica'],
            count_percent['group1']['Iris-virginica'],
            count_percent['group2']['Iris-virginica']] == [30, 50, 20]

    assert result['out'].equals(tst)


def test_perfect_split_k_fold_stratified_success():
    # Even more balanced percentage of each class in each group
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:30, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 1, 'random_state': 0,
                       'column': ['class']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0,
            1, 1, 1, 2, 2, 2, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group1']['Iris-setosa'],
            count_percent['group2']['Iris-setosa']] == [40, 30, 30]

    assert [count_percent['group0']['Iris-versicolor'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group2']['Iris-versicolor']] == [30, 40, 30]

    assert [count_percent['group0']['Iris-virginica'],
            count_percent['group1']['Iris-virginica'],
            count_percent['group2']['Iris-virginica']] == [30, 30, 40]

    assert result['out'].equals(tst)


def test_perfect_split_k_fold_shuffle_stratified_success():
    # Same as perfect_stratified, shuffle doesn't make a difference if stratified
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:30, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 1, 'attribute': 'groups',
                       'stratified': 1, 'random_state': 0,
                       'column': ['class']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            0, 2, 1, 2, 0, 1, 2, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 2, 1, 0, 1, 2,
            1, 1, 0, 0, 2, 2, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group1']['Iris-setosa'],
            count_percent['group2']['Iris-setosa']] == [40, 30, 30]

    assert [count_percent['group0']['Iris-versicolor'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group2']['Iris-versicolor']] == [30, 40, 30]

    assert [count_percent['group0']['Iris-virginica'],
            count_percent['group1']['Iris-virginica'],
            count_percent['group2']['Iris-virginica']] == [30, 30, 40]

    assert result['out'].equals(tst)


def test_unbalanced_split_k_fold_success():
    # Unbalanced example/test, versicolor is occupying almost every groups space
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[8:27, 'class'] = 'Iris-versicolor'
    df[1].loc[27:29, 'class'] = 'Iris-virginica'
    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    data = {
        "groups": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor']] == [80, 20]

    assert count_percent['group1']['Iris-versicolor'] == 100

    assert [count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == [70, 30]

    assert result['out'].equals(tst)


def test_unbalanced_split_k_fold_shuffle_success():
    # Unbalanced example/test, versicolor is occupying almost every groups space
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[8:27, 'class'] = 'Iris-versicolor'
    df[1].loc[27:29, 'class'] = 'Iris-virginica'
    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 1, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    data = {
        "groups": [
            2, 1, 0, 2, 1, 1, 1, 2, 1, 2, 0, 0, 2, 0, 1, 2, 1, 0, 2, 2, 1, 2, 0,
            1, 0, 2, 0, 0, 0, 1
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor'],
            count_percent['group0']['Iris-virginica']] == [10, 70, 20]

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group1']['Iris-virginica']] == [40, 50, 10]

    assert [count_percent['group2']['Iris-setosa'],
            count_percent['group2']['Iris-versicolor']] == [30, 70]

    assert result['out'].equals(tst)


def test_unbalanced_split_k_fold_stratified_success():
    # Stratified does it best to balance it out
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[8:27, 'class'] = 'Iris-versicolor'
    df[1].loc[27:29, 'class'] = 'Iris-virginica'
    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 1, 'column': ['class'], 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    data = {
        "groups": [
            0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 0, 1, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor'],
            count_percent['group0']['Iris-virginica']] == [30, 60, 10]

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group1']['Iris-virginica']] == [30, 60, 10]

    assert [count_percent['group2']['Iris-setosa'],
            count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == [20, 70, 10]

    assert result['out'].equals(tst)


def test_unbalanced_split_k_fold_shuffle_stratified_success():
    # Same as unbalanced_stratified, shuffle doesn't make a difference if stratified
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[8:27, 'class'] = 'Iris-versicolor'
    df[1].loc[27:29, 'class'] = 'Iris-virginica'
    arguments = {
        'parameters': {'n_splits': 3, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 1, 'column': ['class'], 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    data = {
        "groups": [
            0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 0, 1, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 3)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor'],
            count_percent['group0']['Iris-virginica']] == [30, 60, 10]

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group1']['Iris-virginica']] == [30, 60, 10]

    assert [count_percent['group2']['Iris-setosa'],
            count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == [20, 70, 10]

    assert result['out'].equals(tst)


def test_uneven_split_k_fold_success():
    # With a uneven split, you get floating point class presence percentage
    # in some groups
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:29, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 4, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 4)

    assert count_percent['group0']['Iris-setosa'] == 100

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor']] == [25, 75]
    # Here
    assert [count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == pytest.approx(
        [57.14, 42.85], 0.1)

    assert count_percent['group3']['Iris-virginica'] == 100

    assert result['out'].equals(tst)


def test_uneven_split_k_fold_shuffle_success():
    # Same as before
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:29, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 4, 'shuffle': 1, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            3, 2, 0, 3, 2, 1, 2, 3, 1, 2, 0, 0, 3, 0, 1, 3, 1, 1, 2, 2, 1, 3, 1,
            1, 0, 3, 0, 0, 0, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 4)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor'],
            count_percent['group0']['Iris-virginica']] == [12.5, 37.5, 50]

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group1']['Iris-virginica']] == [25, 37.5, 37.5]

    assert [count_percent['group2']['Iris-setosa'],
            count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == pytest.approx(
        [57.14, 28.57, 14.28], 0.1)

    assert [count_percent['group3']['Iris-setosa'],
            count_percent['group3']['Iris-versicolor'],
            count_percent['group3']['Iris-virginica']] == pytest.approx(
        [42.85, 28.57, 28.57], 0.1)

    assert result['out'].equals(tst)


def test_uneven_split_k_fold_stratified_success():
    # Stratified does it's best to balance the uneven split
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:29, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 4, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 1, 'column': ['class'], 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0,
            1, 1, 1, 2, 2, 3, 3
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 4)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor'],
            count_percent['group0']['Iris-virginica']] == [37.5, 25, 37.5]

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group1']['Iris-virginica']] == [37.5, 25, 37.5]

    assert [count_percent['group2']['Iris-setosa'],
            count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == pytest.approx(
        [28.57, 42.85, 28.57], 0.1)

    assert [count_percent['group3']['Iris-setosa'],
            count_percent['group3']['Iris-versicolor'],
            count_percent['group3']['Iris-virginica']] == pytest.approx(
        [28.57, 42.85, 28.57], 0.1)

    assert result['out'].equals(tst)


def test_uneven_split_k_fold_shuffle_stratified_success():
    # Same as uneven_stratified
    df = ['df', util.iris(['class'], 30)]
    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:29, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 4, 'shuffle': 1, 'attribute': 'groups',
                       'stratified': 1, 'column': ['class'], 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SplitKFoldOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    data = {
        "groups": [
            0, 3, 1, 3, 0, 2, 2, 1, 0, 1, 1, 2, 0, 1, 3, 3, 0, 2, 3, 2, 0, 1, 3,
            1, 1, 0, 0, 2, 3, 2
        ]
    }
    tst = pd.concat([df[1], pd.DataFrame(data)], axis=1)
    count_percent = percent(result['out'], 4)

    assert [count_percent['group0']['Iris-setosa'],
            count_percent['group0']['Iris-versicolor'],
            count_percent['group0']['Iris-virginica']] == [37.5, 25, 37.5]

    assert [count_percent['group1']['Iris-setosa'],
            count_percent['group1']['Iris-versicolor'],
            count_percent['group1']['Iris-virginica']] == [37.5, 25, 37.5]

    assert [count_percent['group2']['Iris-setosa'],
            count_percent['group2']['Iris-versicolor'],
            count_percent['group2']['Iris-virginica']] == pytest.approx(
        [28.57, 42.85, 28.57], 0.1)

    assert [count_percent['group3']['Iris-setosa'],
            count_percent['group3']['Iris-versicolor'],
            count_percent['group3']['Iris-virginica']] == pytest.approx(
        [28.57, 42.85, 28.57], 0.1)

    assert result['out'].equals(tst)


def test_split_k_fold_bad_n_split_param_success():
    df = ['df', util.iris(['class'], 30)]

    df[1].loc[10:20, 'class'] = 'Iris-versicolor'
    df[1].loc[20:30, 'class'] = 'Iris-virginica'

    arguments = {
        'parameters': {'n_splits': 1, 'shuffle': 0, 'attribute': 'groups',
                       'stratified': 0, 'random_state': 0},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = SplitKFoldOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert "Parameter 'n_splits' must be x>=2 for task" in str(val_err)
