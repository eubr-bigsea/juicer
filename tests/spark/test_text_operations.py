# -*- coding: utf-8 -*-
from __future__ import absolute_import

import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.spark.text_operation import TokenizerOperation, \
    RemoveStopWordsOperation, \
    WordToVectorOperation, GenerateNGramsOperation

from tests import compare_ast, format_code_comparison


# Test TokenizerOperation

def test_tokenizer_operation_type_simple_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'simple',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col'],
        TokenizerOperation.ALIAS_PARAM: 'c'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = TokenizerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = {3}
            pattern_exp = r'\s+'
            min_token_length = 3
            tokenizers = [RegexTokenizer(inputCol=col, outputCol=alias,
                    pattern=pattern_exp, minTokenLength=min_token_length)
                    for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=tokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(params[TokenizerOperation.ATTRIBUTES_PARAM],
                   n_in['input data'],
                   n_out['output data'],
                   json.dumps(
                       list(zip(params[TokenizerOperation.ATTRIBUTES_PARAM],
                            params[TokenizerOperation.ALIAS_PARAM])))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_tokenizer_operation_type_regexp_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'regex',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col'],
        TokenizerOperation.ALIAS_PARAM: 'c',
        TokenizerOperation.MINIMUM_SIZE: 3,
        TokenizerOperation.EXPRESSION_PARAM: r'\s+'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = TokenizerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = {3}
            pattern_exp = r'{4}'
            min_token_length = {5}

            tokenizers = [RegexTokenizer(inputCol=col, outputCol=alias,
                                pattern=pattern_exp,
                                minTokenLength=min_token_length)
                                for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=tokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(params[TokenizerOperation.TYPE_PARAM], n_in['input data'],
                   n_out['output data'],
                   json.dumps(
                       list(zip(params[TokenizerOperation.ATTRIBUTES_PARAM],
                                params[TokenizerOperation.ALIAS_PARAM]))),
                   params[TokenizerOperation.EXPRESSION_PARAM],
                   params[TokenizerOperation.MINIMUM_SIZE]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_stopwords_operations_2_params_success():
    case_sensitive = 'False'
    n_in = {'input data': 'input_1', 'stop words': 'stops_pt'}
    n_out = {'output data': 'output_1'}

    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: ['text'],
        RemoveStopWordsOperation.ALIAS_PARAM: 'c',
        RemoveStopWordsOperation.STOP_WORD_LIST_PARAM: n_in['stop words'],
        RemoveStopWordsOperation.STOP_WORD_ATTRIBUTE_PARAM: 'stop_word',
        RemoveStopWordsOperation.STOP_WORD_CASE_SENSITIVE_PARAM: case_sensitive
    }
    # Input data, and StopWords list

    instance = RemoveStopWordsOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        sw = [stop[0].strip() for stop in {}.collect() if stop and stop[0]]
        """.format(params[RemoveStopWordsOperation.STOP_WORD_LIST_PARAM]))

    expected_code += dedent("""
        col_alias = {3}
        case_sensitive = {4}
        removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                    stopWords=sw, caseSensitive=case_sensitive)
                    for col, alias in col_alias]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=removers)
        {2} = pipeline.fit({1}).transform({1})
        """).format(params[RemoveStopWordsOperation.ATTRIBUTES_PARAM],
                    n_in['input data'],
                    n_out['output data'],
                    json.dumps(
                        list(zip(
                            params[RemoveStopWordsOperation.ATTRIBUTES_PARAM],
                            params[RemoveStopWordsOperation.ALIAS_PARAM]))),
                    case_sensitive
                    )

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_stopwords_operations_1_params_success():
    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: ['text'],
        RemoveStopWordsOperation.ALIAS_PARAM: 'c',
        RemoveStopWordsOperation.STOP_WORD_LIST_PARAM: 'stop_word_list',
        RemoveStopWordsOperation.STOP_WORD_LANGUAGE_PARAM: 'english',
        RemoveStopWordsOperation.STOP_WORD_CASE_SENSITIVE_PARAM: 'False'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = RemoveStopWordsOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    expected_code = ''
    code = instance.generate_code()

    if len(n_in) != 2:
        expected_code = 'sw = ["stop_word_list"]'

    expected_code += dedent("""
        col_alias = {3}
        case_sensitive = {4}
        removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                    stopWords=sw, caseSensitive=case_sensitive)
                    for col, alias in col_alias]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=removers)
        {2} = pipeline.fit({1}).transform({1})
        """.format(
        params[RemoveStopWordsOperation.ATTRIBUTES_PARAM],
        n_in['input data'],
        n_out['output data'],
        json.dumps(
            list(zip(params[RemoveStopWordsOperation.ATTRIBUTES_PARAM],
                     params[RemoveStopWordsOperation.ALIAS_PARAM]))),
        params[RemoveStopWordsOperation.STOP_WORD_CASE_SENSITIVE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


# Test WordToVectorOperation

# Typecount
def test_word_to_vector_count_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'count',
        WordToVectorOperation.ATTRIBUTES_PARAM: ['text'],
        WordToVectorOperation.ALIAS_PARAM: 'c',
        WordToVectorOperation.VOCAB_SIZE_PARAM: 6,
        WordToVectorOperation.MINIMUM_DF_PARAM: 5,
        WordToVectorOperation.MINIMUM_TF_PARAM: 4
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1', 'vocabulary': 'vocab_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
                col_alias = {3}
                vectorizers = [CountVectorizer(minTF={4}, minDF={5},
                               vocabSize={6}, binary=False, inputCol=col,
                               outputCol=alias) for col, alias in col_alias]
                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=vectorizers)
                vector_model_1 = pipeline.fit({1})
                {2} = vector_model_1.transform({1})

                {7} = dict([(col_alias[i][1], v.vocabulary)
                        for i, v in enumerate(vector_model_1.stages)])
                """.format(params[WordToVectorOperation.ATTRIBUTES_PARAM],
                           n_in['input data'],
                           n_out['output data'],
                           json.dumps(list(zip(
                               params[WordToVectorOperation.ATTRIBUTES_PARAM],
                               params[WordToVectorOperation.ALIAS_PARAM]))),
                           params[WordToVectorOperation.MINIMUM_TF_PARAM],
                           params[WordToVectorOperation.MINIMUM_DF_PARAM],
                           params[WordToVectorOperation.VOCAB_SIZE_PARAM],
                           n_out['vocabulary']

                           ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


# Check requirements
def test_word_to_vector_word2vec_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'word2vec',
        WordToVectorOperation.ATTRIBUTES_PARAM: ['text'],
        WordToVectorOperation.ALIAS_PARAM: 'c',
        WordToVectorOperation.MINIMUM_COUNT_PARAM: 3,
        WordToVectorOperation.MINIMUM_VECTOR_SIZE_PARAM: 0
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1', 'vocabulary': 'vocab_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    # @FIXME Check

    expected_code = dedent("""
                col_alias = {3}
                vectorizers = [Word2Vec(vectorSize={4}, minCount={5},
                            numPartitions=1,
                            stepSize=0.025,
                            maxIter=1,
                            seed=None,
                            inputCol=col,
                            outputCol=alias
                            ) for col, alias in col_alias]
                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=vectorizers)
                vector_model_1 = pipeline.fit({1})
                {2} = vector_model_1.transform({1})

                {6} = dict([(col_alias[i][1], v.getVectors())
                        for i, v in enumerate(vector_model_1.stages)])
                """.format(
        params[WordToVectorOperation.ATTRIBUTES_PARAM],
        n_in['input data'],
        n_out['output data'],
        json.dumps(list(zip(
            params[WordToVectorOperation.ATTRIBUTES_PARAM],
            params[WordToVectorOperation.ALIAS_PARAM]))),
        params[
            WordToVectorOperation.MINIMUM_VECTOR_SIZE_PARAM],
        params[WordToVectorOperation.MINIMUM_COUNT_PARAM],
        n_out['vocabulary']))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


# Test NGramOperations
def test_n_gram_operations_success():
    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: ['text'],
        GenerateNGramsOperation.ALIAS_PARAM: 'c',
        GenerateNGramsOperation.N_PARAM: '2'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = GenerateNGramsOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    # NGram(self, n=2, inputCol=None, outputCol=None
    expected_code = dedent("""
            col_alias = {3}
            n_gramers = [NGram(n={0}, inputCol=col,
                           outputCol=alias) for col, alias in col_alias]
            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=n_gramers)
            model = pipeline.fit({1})
            {2} = model.transform({1})
            """.format(params[GenerateNGramsOperation.N_PARAM],
                       n_in['input data'], n_out['output data'],
                       json.dumps(
                           list(zip(
                               params[GenerateNGramsOperation.ATTRIBUTES_PARAM],
                               params[GenerateNGramsOperation.ALIAS_PARAM])))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


# Test NGramOperations
def test_n_gram_missing_param_failure():
    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: ['text'],
        GenerateNGramsOperation.ALIAS_PARAM: 'c'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        GenerateNGramsOperation(params, named_inputs=n_in,
                                named_outputs=n_out)
