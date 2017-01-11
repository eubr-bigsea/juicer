# -*- coding: utf-8 -*-
import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.spark.text_operation import TokenizerOperation, RemoveStopWordsOperation, \
    WordToVectorOperation, GenerateNGramsOperation

from tests import compare_ast, format_code_comparison


def debug_ast(code, expected_code):
    print
    print code
    print '*' * 20
    print expected_code
    print '*' * 20


# Test TokenizerOperation

def test_tokenizer_operation_type_simple_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'simple',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col'],
        TokenizerOperation.ALIAS_PARAM: 'c'
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = TokenizerOperation(params, inputs, outputs,
                                  named_inputs={}, named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = {3}
            tokenizers = [Tokenizer(inputCol=col, outputCol=alias)
                                for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=tokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(params[TokenizerOperation.ATTRIBUTES_PARAM], inputs[0], outputs[0],
                   json.dumps(zip(params[TokenizerOperation.ATTRIBUTES_PARAM],
                                  params[TokenizerOperation.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)

# @FIXME
def test_tokenizer_operation_type_regexp_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'regex',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col'],
        TokenizerOperation.ALIAS_PARAM: 'c',
        TokenizerOperation.MINIMUM_SIZE: 3,
        TokenizerOperation.EXPRESSION_PARAM: r'\s+'
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = TokenizerOperation(params, inputs, outputs,
                                  named_inputs={}, named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = {3}
            pattern = {4}
            min_token_length = {5}

            regextokenizers = [RegexTokenizer(inputCol=col, outputCol=alias)
                                for col, alias in col_alias], pattern=pattern,
                                 minTokenLength=min_token_length)

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=regextokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(params[TokenizerOperation.ATTRIBUTES_PARAM], inputs, outputs,
                   json.dumps(zip(params[TokenizerOperation.ATTRIBUTES_PARAM],
                                  params[TokenizerOperation.ALIAS_PARAM])),
                   params[TokenizerOperation.EXPRESSION_PARAM],
                   params[TokenizerOperation.MINIMUM_SIZE]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + debug_ast(code, expected_code)


def test_remove_stopwords_operations_2_params_success():
    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: 'text',
        RemoveStopWordsOperation.ALIAS_PARAM: 'col_alias',
        RemoveStopWordsOperation.STOP_WORD_LIST_PARAM: 'df_2',
        RemoveStopWordsOperation.STOP_WORD_ATTRIBUTE_PARAM: 'stop_word'
    }
    # Input data, and StopWords list
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance = RemoveStopWordsOperation(params, inputs, outputs,
                        named_inputs={'input data': 'df_1', 'stop words': 'df_2'},
                        named_outputs={})

    code = instance.generate_code()

    expected_code = "sw = [stop[0].strip() for stop in {}.collect()]".format(
        params[RemoveStopWordsOperation.STOP_WORD_LIST_PARAM])

    expected_code += dedent("""
        col_alias = {3}
        removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                    stopWords=sw)for col, alias in col_alias]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=removers)
        {2} = pipeline.fit({1}).transform({1})
        """).format(params[RemoveStopWordsOperation.ATTRIBUTES_PARAM], inputs[0],
                   outputs,
                   json.dumps(zip(params[RemoveStopWordsOperation.ATTRIBUTES_PARAM],
                                params[RemoveStopWordsOperation.ALIAS_PARAM]))
                   )

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_remove_stopwords_operations_1_params_success():
    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: 'text',
        RemoveStopWordsOperation.ALIAS_PARAM: 'col_alias',
        RemoveStopWordsOperation.STOP_WORD_LANGUAGE: 'english'
    }
    inputs = ['df_1']
    outputs = ['output_1']

    instance = RemoveStopWordsOperation(params, inputs, outputs,
                                        named_inputs={'input data': 'df_1'},
                                        named_outputs={'output data': 'df_3'})

    code = instance.generate_code()

    expected_code = "sw = StopWordsRemover.loadDefaultStopWords({})".format(
        params[RemoveStopWordsOperation.STOP_WORD_LANGUAGE])
    #
    # if len(inputs) != 2:
    #     expected_code = "sw = {}".format(json.dumps
    # (params[RemoveStopWordsOperation.STOP_WORD_LIST_PARAM]))
    # else:
    #     expected_code = "sw = [stop[0].strip() for stop in {}.collect()]".
    # format(named_inputs['stop words'])

    expected_code += dedent("""
        col_alias = {3}
        removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                    stopWords=sw)for col, alias in col_alias]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=removers)
        {2} = pipeline.fit({1}).transform({1})
        """.format(params[RemoveStopWordsOperation.ATTRIBUTES_PARAM], inputs[0],
                   outputs,
                   json.dumps(zip(params[RemoveStopWordsOperation.ATTRIBUTES_PARAM],
                                params[RemoveStopWordsOperation.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


# Test WordToVectorOperation

# Typecount
def test_word_to_vector_count_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'count',
        WordToVectorOperation.ATTRIBUTES_PARAM: 'text',
        WordToVectorOperation.ALIAS_PARAM: 'col_alias',
        WordToVectorOperation.VOCAB_SIZE_PARAM: '6',
        WordToVectorOperation.MINIMUM_DF_PARAM: '5',
        WordToVectorOperation.MINIMUM_TF_PARAM: '4'
    }

    inputs = ['df_1']
    outputs = ['output_1']

    instance = WordToVectorOperation(params, inputs, outputs,
                                     named_inputs={},
                                     named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
                col_alias = {3}
                vectorizers = [CountVectorizer(minTF={4}, minDF={5},
                               vocabSize={6}, binary=False, inputCol=col,
                               outputCol=alias) for col, alias in col_alias]
                """.format(params[WordToVectorOperation.ATTRIBUTES_PARAM], inputs[0],
                        outputs[0],
                        json.dumps(zip(params[WordToVectorOperation.ATTRIBUTES_PARAM],
                                        params[WordToVectorOperation.ALIAS_PARAM])),
                        params[WordToVectorOperation.MINIMUM_TF_PARAM],
                        params[WordToVectorOperation.MINIMUM_DF_PARAM],
                        params[WordToVectorOperation.VOCAB_SIZE_PARAM]
                        ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


# Check requirements
def test_word_to_vector_word2vec_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'word2vec',
        WordToVectorOperation.ATTRIBUTES_PARAM: 'text',
        WordToVectorOperation.ALIAS_PARAM: 'col_alias',
        WordToVectorOperation.VOCAB_SIZE_PARAM: '6',
        WordToVectorOperation.MINIMUM_DF_PARAM: '5',
        WordToVectorOperation.MINIMUM_TF_PARAM: '4'
    }

    inputs = ['df_1']
    outputs = ['output_1']

    instance = WordToVectorOperation(params, inputs, outputs,
                                     named_inputs={},
                                     named_outputs={})

    code = instance.generate_code()
    # @FIXME Implement
    # Word2Vec(self, vectorSize=100, minCount=5, numPartitions=1, stepSize=0.025,
    # maxIter=1, seed=None,
    # inputCol=None, outputCol=None, windowSize=5, maxSentenceLength=1000)[source]
    expected_code = dedent("""
                col_alias = {3}
                vectorizers = [Word2Vec(vectorSize={4}, minCount=2,
                            inputCol=col,
                            outputCol=alias) for col, alias in col_alias]

                """.format(params[WordToVectorOperation.ATTRIBUTES_PARAM], inputs[0],
                        outputs,
                        json.dumps(zip(params[WordToVectorOperation.ATTRIBUTES_PARAM],
                                      params[WordToVectorOperation.ALIAS_PARAM])),
                        params[WordToVectorOperation.VOCAB_SIZE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


# Test NGramOperations
def test_n_gram_operations_success():

    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: 'text',
        GenerateNGramsOperation.ALIAS_PARAM: 'col_alias',
        GenerateNGramsOperation.N_PARAM: '2'
    }
    inputs = ['df_1']
    outputs = ['output_1']

    instance = GenerateNGramsOperation(params, inputs, outputs,
                                       named_inputs={},
                                       named_outputs={})

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
                    inputs, outputs,
                    json.dumps(zip(params[GenerateNGramsOperation.ATTRIBUTES_PARAM],
                                    params[GenerateNGramsOperation.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg

    # Test NGramOperations


def test_n_gram_operations_failure():
    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: 'text',
        GenerateNGramsOperation.ALIAS_PARAM: 'col_alias',
        GenerateNGramsOperation.N_PARAM: '-1'
    }
