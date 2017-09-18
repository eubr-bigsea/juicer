# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest

class TokenizerOperation(Operation):
    """
    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.
    """

    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'
    MINIMUM_SIZE = 'min_token_length'

    TYPE_SIMPLE = 'simple'
    TYPE_REGEX = 'regex'


    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
        if self.type not in [self.TYPE_REGEX, self.TYPE_SIMPLE]:
            raise ValueError(
                'Invalid type for operation Tokenizer: {}'.format(self.type))
        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) > 0
        self.has_import = "from functions.text.Tokenizer import TokenizerOperation\n"



    def generate_code(self):
        input_data  = self.named_inputs['input data']
        self.output = self.named_outputs['output data']


        if self.type == self.TYPE_SIMPLE:
            code =  """
                        numFrag = 4
                        settings = dict()
                        settings['min_token_length'] = {min_token_length}
                        settings['type'] = 'simple'
                        settings['attributes'] = {att}
                        settings['alias'] = {alias}
                        {output} = TokenizerOperation({input},settings,numFrag)

                    """.format(output           = self.output,
                               min_token_length = self.min_token_lenght,
                               input            = input_data,
                               att = self.attributes,
                               alias = self.alias[0]
                               )

        return dedent(code)

class RemoveStopWordsOperation(Operation):
    """
    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don’t carry
    as much meaning.
    """

    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    STOP_WORD_LIST_PARAM = 'stop_word_list'
    STOP_WORD_ATTRIBUTE_PARAM = 'stop_word_attribute'
    STOP_WORD_LANGUAGE_PARAM = 'stop_word_language'
    STOP_WORD_CASE_SENSITIVE_PARAM = 'sw_case_sensitive'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.stop_word_attribute = self.parameters.get(
            self.STOP_WORD_ATTRIBUTE_PARAM, '')


        tmp = [s.strip() for s in
               self.parameters.get(self.STOP_WORD_LIST_PARAM,'').split(',')]
        self.stop_word_list = tmp if tmp != [u''] else []


        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.stop_word_language = self.parameters.get(
            self.STOP_WORD_LANGUAGE_PARAM, 'english')

        self.sw_case_sensitive = self.parameters.get(
            self.STOP_WORD_CASE_SENSITIVE_PARAM, 'False')





        self.stopwords_list  = parameters.get(self.STOP_WORD_LIST_PARAM, [])

        self.stopwords_input =  self.named_inputs.get('stop words', [])
        self.output = self.named_outputs['output data']


        self.has_import = "from functions.text.RemoveStopWords import RemoveStopWordsOperation\n"
        self.has_code = len(self.named_inputs) > 0

        #self.generate_code()


    def generate_code(self):

        code =  """
                numFrag = 4
                settings = dict()
                settings['attribute'] = {att}
                settings['alias']     = {alias}
                settings['attribute-stopwords'] = '{att_stop}'
                settings['case-sensitive']      = {case}
                settings['news-stops-words']    = {stopwords_list}

                {output}=RemoveStopWordsOperation({input},settings,{stopwords_input},numFrag)
                """.format(
                            att      = self.attributes,
                            att_stop = self.stop_word_attribute,
                            alias    = self.alias,
                            case     = self.sw_case_sensitive,

                            output = self.output,
                            input  = self.named_inputs['input data'],

                           stopwords_list = self.stop_word_list,
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


        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000)
        if self.vocab_size == "":
            self.vocab_size = -1

        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 5)
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1)

        self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM, 3)
        self.minimum_count = parameters.get(self.MINIMUM_COUNT_PARAM, 0)

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)

        if self.type == 'count':
            self.type = 'TF-IDF'
        else:
            self.type = "BoW"

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')]

        self.input_data = self.named_inputs['input data']

        self.vocab = self.named_outputs.get('vocabulary', 'tmp')
        self.output = self.named_outputs['output data']
        self.has_code = len(self.named_inputs) > 0
        self.has_import = "from functions.text.ConvertWordstoVector import ConvertWordstoVectorOperation\n"
        self.generate_code()


    def generate_code(self):


        code =  """
                numFrag = 4
                params = dict()
                params['attributes'] = {attrib}
                params['alias']      = '{alias}'
                params['minimum_tf'] = {tf}
                params['minimum_df'] = {df}
                params['size']       = {size}
                params['mode']       = '{mode}'
                {output}, {vocabulary} = ConvertWordstoVectorOperation({input}, params, numFrag)
                """.format(output    = self.output,
                           vocabulary= self.vocab,
                           input = self.input_data,
                           size  = self.vocab_size,
                           df    = self.minimum_df,
                           tf    = self.minimum_tf,
                           attrib = self.attributes,
                           alias  = self.alias[0],
                           mode  = self.type
                           )


        return dedent(code)