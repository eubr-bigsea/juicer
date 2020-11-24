from tests.scikit_learn import util
from juicer.scikit_learn.text_operation import GenerateNGramsOperation
import pandas as pd
from nltk.util import ngrams
import pytest

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# GenerateNGrams
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_generate_n_grams_success():
    df = util.iris(['class'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 1},
                       'n': 3},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GenerateNGramsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    grams = []
    for row in test_out['class'].to_numpy():
        grams.append([" ".join(gram) for gram in ngrams(row, 3)])

    test_out['class_ngram'] = grams
    assert result['out'].equals(test_out)


def test_generate_n_grams_alias_param_success():
    df = util.iris(['class'], size=10)

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 1},
                       'n': 1,
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GenerateNGramsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].columns[1] == 'success'


def test_generate_n_grams_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 1},
                       'n': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = GenerateNGramsOperation(**arguments)
    assert instance.generate_code() is None


def test_generate_n_grams_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 1},
                       'n': 1},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GenerateNGramsOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_generate_n_grams_missing_attributes_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'input data': 1},
                       'n': 3},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GenerateNGramsOperation(**arguments)
    assert "Parameter 'attributes' must be informed for task" \
           in str(val_err.value)


def test_generate_n_grams_missing_n_param_fail():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 1}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GenerateNGramsOperation(**arguments)
    assert "Parameter 'n' must be informed for task" \
           in str(val_err.value)


def test_generate_n_grams_missing_multiplicity_param_fail():
    df = util.iris(['class'], size=10)

    arguments = {
        'parameters': {'attributes': ['class'],
                       'n': 3},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GenerateNGramsOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "'multiplicity'" in str(key_err.value)
