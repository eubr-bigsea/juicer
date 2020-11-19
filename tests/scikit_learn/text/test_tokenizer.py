from tests.scikit_learn import util
from juicer.scikit_learn.text_operation import TokenizerOperation
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import regexp_tokenize
import pandas as pd
import pytest

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Tokenizer
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_tokenizer_type_simple_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:5, ['class']] = 'Iris-versicolor'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    test_result = []
    toktok = TweetTokenizer()

    for row in test_out['class'].to_numpy():
        test_result.append([word
                            for word in toktok.tokenize(row) if len(word) >= 3])
    test_out['class_tok'] = test_result

    assert result['out'].equals(test_out)


def test_tokenizer_type_simple_alias_param_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:5, ['class']] = 'Iris-versicolor'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    test_result = []
    toktok = TweetTokenizer()

    for row in test_out['class'].to_numpy():
        test_result.append([word
                            for word in toktok.tokenize(row) if len(word) >= 3])
    test_out['success'] = test_result

    assert result['out'].equals(test_out)


def test_tokenizer_type_simple_minimum_token_length_param_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:5, ['class']] = 'Iris-versicolor'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'min_token_length': 15},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    from nltk.tokenize import TweetTokenizer
    test_out = df
    test_result = []
    toktok = TweetTokenizer()

    for row in test_out['class'].to_numpy():
        test_result.append([word
                            for word in toktok.tokenize(row) if len(word) >= 15])
    test_out['class_tok'] = test_result
    assert result['out'].equals(test_out)


def test_tokenizer_type_regex_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:4, ['class']] = 'Iris-versicolor'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'type': 'regex'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    test_result = []

    for row in test_out['class'].to_numpy():
        test_result.append([word for word in
                            regexp_tokenize(row, pattern='\s+') if
                            len(word) >= 3])

    test_out['class_tok'] = test_result
    assert result['out'].equals(test_out)


def test_tokenizer_type_regex_alias_param_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:4, ['class']] = 'Iris-versicolor'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'type': 'regex', 'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    test_result = []

    for row in test_out['class'].to_numpy():
        test_result.append([word for word in
                            regexp_tokenize(row, pattern='\s+') if
                            len(word) >= 3])

    test_out['success'] = test_result
    assert result['out'].equals(test_out)


def test_tokenizer_type_regex_minimum_token_length_param_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:4, ['class']] = 'Iris-versicolor'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'type': 'regex', 'min_token_length': 15},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    test_result = []

    for row in test_out['class'].to_numpy():
        test_result.append([word for word in
                            regexp_tokenize(row, pattern='\s+') if
                            len(word) >= 15])

    test_out['class_tok'] = test_result
    assert result['out'].equals(test_out)


def test_tokenizer_type_regex_expression_param_success():
    df = util.iris(['class'], size=10)
    df.loc[0:2, ['class']] = 'Iris-virginica'
    df.loc[3:4, ['class']] = 'Iris-versicolor'
    df.loc[5, ['class']] = 'Not-setosa'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'type': 'regex', 'expression': 'Iris-\w+'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    test_result = []

    for row in test_out['class'].to_numpy():
        test_result.append([word for word in
                            regexp_tokenize(row, pattern='Iris-\w+') if
                            len(word) >= 3])

    test_out['class_tok'] = test_result
    assert result['out'].equals(test_out)


def test_tokenizer_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = TokenizerOperation(**arguments)
    assert instance.generate_code() is None


def test_tokenizer_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       },
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_tokenizer_missing_attributes_param_fail():
    arguments = {
        'parameters': {
            'multiplicity': {'input data': 0},
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        TokenizerOperation(**arguments)
    assert "Parameter 'attributes' must be informed for task" in str(
        val_err.value)


def test_tokenizer_invalid_type_param_fail():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'type': 'invalid'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        TokenizerOperation(**arguments)
    assert "Invalid type for operation Tokenizer: invalid" in str(
        val_err.value)


def test_tokenizer_missing_multiplicity_param_fail():
    df = util.iris(['class'], size=10)
    arguments = {
        'parameters': {'attributes': ['class'],
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TokenizerOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "'multiplicity'" in str(key_err.value)
