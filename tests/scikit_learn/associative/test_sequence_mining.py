from tests.scikit_learn import util
from textwrap import dedent
from juicer.scikit_learn.associative_operation import SequenceMiningOperation
from prefixspan import PrefixSpan
import pytest
import pandas as pd


# TODO: tests using/comparing with a well-known result
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# SequenceMining
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_sequence_mining_success():
    db = [
        [[0, 1, 2, 3, 4]],
        [[1, 1, 1, 3, 4]],
        [[2, 1, 2, 2, 0]],
        [[1, 1, 1, 2, 2]],
    ]
    df = pd.DataFrame(db, columns=['transactions'])
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 0.1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SequenceMiningOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    transactions = test_df[test_df.columns[0]].to_numpy().tolist()
    min_support = 0.1 * len(transactions)
    span = PrefixSpan(transactions)
    span.minlen, span.maxlen = 1, 10
    test_result = span.frequent(min_support)
    test_df = pd.DataFrame(test_result, columns=['support', 'itemsets'])

    assert result['out'].equals(test_df)
    assert dedent("""
    transactions = df['df.columns[0]'].to_numpy().tolist() 
    min_support = 0.1 * len(transactions)
    
    class PrefixSpan2(PrefixSpan):
        def __init__(self, db, minlen=1, maxlen=1000):
            self._db = db
            self.minlen, self.maxlen = minlen, maxlen
            self._results: Any = []
    
    span = PrefixSpan2(transactions, minlen=1, maxlen=10)
    result = span.frequent(min_support, closed=False, generator=False)
    
    out = pd.DataFrame(result, columns=['support', 'itemsets'])
    """) == instance.generate_code()


def test_sequence_mining_attribute_param_success():
    db = [
        [[0, 1, 2, 3, 4]],
        [[1, 1, 1, 3, 4]],
        [[2, 1, 2, 2, 0]],
        [[1, 1, 1, 2, 2]],
    ]
    df = pd.DataFrame(db, columns=['transactions'])
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 0.2,
                       'attribute': ['transactions']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SequenceMiningOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    transactions = test_df['transactions'].to_numpy().tolist()
    min_support = 0.2 * len(transactions)
    span = PrefixSpan(transactions)
    span.minlen, span.maxlen = 1, 10
    test_result = span.frequent(min_support)
    test_df = pd.DataFrame(test_result, columns=['support', 'itemsets'])

    assert result['out'].equals(test_df)
    assert dedent("""
    transactions = df[''transactions''].to_numpy().tolist() 
    min_support = 0.2 * len(transactions)
    
    class PrefixSpan2(PrefixSpan):
        def __init__(self, db, minlen=1, maxlen=1000):
            self._db = db
            self.minlen, self.maxlen = minlen, maxlen
            self._results: Any = []
    
    span = PrefixSpan2(transactions, minlen=1, maxlen=10)
    result = span.frequent(min_support, closed=False, generator=False)
    
    out = pd.DataFrame(result, columns=['support', 'itemsets'])
    """) == instance.generate_code()


def test_sequence_mining_max_pattern_and_min_support_params_success():
    db = [
        [[0, 1, 2, 3, 4]],
        [[1, 1, 1, 3, 4]],
        [[2, 1, 2, 2, 0]],
        [[1, 1, 1, 2, 2]],
    ]
    df = pd.DataFrame(db, columns=['transactions'])
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 0.5,
                       'max_pattern_length': 3},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SequenceMiningOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    transactions = test_df[test_df.columns[0]].to_numpy().tolist()
    min_support = 0.5 * len(transactions)
    span = PrefixSpan(transactions)
    span.minlen, span.maxlen = 1, 3
    test_result = span.frequent(min_support)
    test_df = pd.DataFrame(test_result, columns=['support', 'itemsets'])
    assert result['out'].equals(test_df)

    assert dedent("""
    transactions = df['df.columns[0]'].to_numpy().tolist() 
    min_support = 0.5 * len(transactions)
    
    class PrefixSpan2(PrefixSpan):
        def __init__(self, db, minlen=1, maxlen=1000):
            self._db = db
            self.minlen, self.maxlen = minlen, maxlen
            self._results: Any = []
    
    span = PrefixSpan2(transactions, minlen=1, maxlen=3)
    result = span.frequent(min_support, closed=False, generator=False)
    
    out = pd.DataFrame(result, columns=['support', 'itemsets'])
    """) == instance.generate_code()


def test_sequence_mining_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'min_support': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = SequenceMiningOperation(**arguments)
    assert instance.generate_code() is None


def test_sequence_mining_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'min_support': 1},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SequenceMiningOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_sequence_mining_missing_min_support_param_fail():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SequenceMiningOperation(**arguments)
    assert f"Parameter 'min_support' must be informed for task" \
           f" {SequenceMiningOperation}" in str(val_err.value)


def test_sequence_mining_invalid_min_support_param_fail():
    arguments = {
        'parameters': {'min_support': -1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SequenceMiningOperation(**arguments)
    assert "Support must be greater or equal to 0.0001 and smaller than 1.0" in \
           str(val_err.value)
