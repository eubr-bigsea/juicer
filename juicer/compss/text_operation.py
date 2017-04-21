# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent
from juicer.operation import Operation


class GenerateNGramsOperation(Operation):
    """ Generates N-Grams from word vectors
    An n-gram is a sequence of n tokens (typically words) for some
    integer n. The NGram class can be used to transform input features
    into n-grams.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)




        self.generate_code()





    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs['output data']

        code = code = "" \
            .format()

        return dedent(code)

class TokenizerOperation(Operation):
    """
    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.
    """

    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    MINIMUM_SIZE = 'min_token_length'

    TYPE_SIMPLE = 'simple'
    TYPE_REGEX = 'regex'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        self.has_code = len(self.named_inputs) > 0
        #self.generate_code()




    def generate_code(self):
        input_data = self.named_inputs['input data']
        self.output = self.named_outputs['output data']

        code =  """
                    numFrag = 4
                    settings = dict()
                    settings['min_token_length'] = {min_token_length}
                    settings['type'] = '{type}'
                    {output} = Tokenizer({input},settings,numFrag)

                """.format(output           = self.output,
                           min_token_length = self.min_token_lenght,
                           type             = self.type,
                           input            = input_data
                           )

        return dedent(code)

class RemoveStopWordsOperation(Operation):
    """
    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don’t carry
    as much meaning.
    """


    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.stopwords_list  = parameters.get('stop_word_list', "[]")
        self.case            = parameters.get('case-sensitive',0) in (1, '1', True)
        self.stopwords_input =  parameters.get('stop words', [])

        self.has_code = len(self.named_inputs) > 0

        #self.generate_code()


    def generate_code(self):
        input_data  = self.named_inputs['input data']
        self.output = self.named_outputs['output data']


        code =  """
                numFrag = 4
                settings = dict()
                settings['news-stops-words'] =  np.array([{stopwords_list}])
                settings['case-sensitive']   = {case}
                {output}=RemoveStopWords({input},settings,{stopwords_input},numFrag)
                """.format(output = self.output,
                           input  = input_data,
                           case   = self.case,
                           stopwords_list = ', '.join(['"{}"'.format(x) for x in self.stopwords_list.split(",")]),
                           stopwords_input = self.stopwords_input
                           )

        return dedent(code)


class WordToVectorOperation(Operation):
    """

    """
    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    VOCAB_SIZE_PARAM = 'vocab_size'
    MINIMUM_DF_PARAM = 'minimum_df'
    MINIMUM_TF_PARAM = 'minimum_tf'

    MINIMUM_VECTOR_SIZE_PARAM = 'minimum_size'
    MINIMUM_COUNT_PARAM = 'minimum_count'

    TYPE_COUNT = 'count'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)
        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000)
        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 5)
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1)

        self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM, 3)
        self.minimum_count = parameters.get(self.MINIMUM_COUNT_PARAM, 0)

        self.has_code = len(self.named_inputs) > 0
        self.generate_code()

    def get_data_out_names(self, sep=','):
        return self.named_outputs['output data']

    def get_output_names(self, sep=", "):
        output = self.named_outputs['output data']
        return sep.join([output, self.named_outputs.get(
            'vocabulary', 'vocab_task_{}'.format(self.order))])

    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs['output data']

        code =  """
                numFrag = 4
                params = dict()
                params['minimum_tf'] = {tf}
                params['minimum_df'] = {df}
                params['size']  =      {size}

                bow         = Bag_of_Words()
                vocabulary  = bow.fit({input},params,numFrag)
                {output}    = bow.transform({input},vocabulary,numFrag)

                """.format(output  = output,
                           input   = input_data,
                           size    = self.vocab_size,
                           df   = self.minimum_df,
                           tf   = self.minimum_tf
                           )


        return dedent(code)