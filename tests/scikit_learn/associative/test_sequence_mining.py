from tests.scikit_learn import util
from juicer.scikit_learn.associative_operation import SequenceMiningOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# SequenceMining
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_sequence_mining_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 1},
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
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == """
transactions = df['df.columns[0]'].to_numpy().tolist() 
min_support = 1.0 * len(transactions)

class PrefixSpan2(PrefixSpan):
    def __init__(self, db, minlen=1, maxlen=1000):
        self._db = db
        self.minlen, self.maxlen = minlen, maxlen
        self._results: Any = []

span = PrefixSpan2(transactions, minlen=1, maxlen=10)
result = span.frequent(min_support, closed=False, generator=False)

out = pd.DataFrame(result, columns=['support', 'itemsets'])
"""


def test_sequence_mining_attribute_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 1,
                       'attribute': ['sepallength']},
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
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == """
transactions = df[''sepallength''].to_numpy().tolist() 
min_support = 1.0 * len(transactions)

class PrefixSpan2(PrefixSpan):
    def __init__(self, db, minlen=1, maxlen=1000):
        self._db = db
        self.minlen, self.maxlen = minlen, maxlen
        self._results: Any = []

span = PrefixSpan2(transactions, minlen=1, maxlen=10)
result = span.frequent(min_support, closed=False, generator=False)

out = pd.DataFrame(result, columns=['support', 'itemsets'])
"""


def test_sequence_mining_max_pattern_length_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 1,
                       'max_pattern_length': 15},
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
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == """
transactions = df['df.columns[0]'].to_numpy().tolist() 
min_support = 1.0 * len(transactions)

class PrefixSpan2(PrefixSpan):
    def __init__(self, db, minlen=1, maxlen=1000):
        self._db = db
        self.minlen, self.maxlen = minlen, maxlen
        self._results: Any = []

span = PrefixSpan2(transactions, minlen=1, maxlen=15)
result = span.frequent(min_support, closed=False, generator=False)

out = pd.DataFrame(result, columns=['support', 'itemsets'])
"""


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
    assert "Parameter 'min_support' must be informed for task" in str(
        val_err.value)


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
