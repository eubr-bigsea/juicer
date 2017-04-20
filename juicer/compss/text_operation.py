# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent
from juicer.operation import Operation

#####from juicer.service import limonero_service
##from juicer.include.metadata import MetadataGet

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

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)




        self.generate_code()




    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs['output data']

        code = ''

        return dedent(code)

class RemoveStopWordsOperation(Operation):
    """
    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don’t carry
    as much meaning.
    """


    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


        self.generate_code()


    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs['output data']

        code = ''

        return dedent(code)
