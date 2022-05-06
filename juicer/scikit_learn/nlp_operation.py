# -*- coding: utf-8 -*-
from textwrap import dedent

import re
from itertools import zip_longest
from juicer.operation import Operation



class TokenizeOperation(Operation):
    """TokenizerOperation.
    The tokenization operation breaks a paragraph of text into smaller pieces, such as words or sentences. Token is a single entity that is
    building blocks for sentence or paragraph.This operation classifying tokens into their parts of speech.
    A simple Tokenizer class provides this functionality.
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

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:

            self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
            if self.type not in [self.TYPE_REGEX, self.TYPE_SIMPLE]:
                raise ValueError(_('Invalid type for '
                                   'operation Tokenizer: {}').format(self.type))

            self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\\s+')

            self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                            self.ATTRIBUTES_PARAM, self.__class__))

            self.alias = [alias.strip() for alias in
                          parameters.get(self.ALIAS_PARAM, '').split(',')]
            # Adjust alias in order to have the same number of aliases as
            # attributes by filling missing alias with the attribute name
            # sufixed by _indexed.
            self.alias = [x[1] or '{}_tok_tag'.format(x[0]) for x in
                          zip_longest(self.attributes,
                                      self.alias[:len(self.attributes)])]

            self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\\s+')
            if len(self.expression_param) == 0:
                self.expression_param = '\\s+'

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            if self.min_token_lenght is not None:
                self.min_token_lenght = \
                    ' if len(word) >= {}'.format(self.min_token_lenght)

            if self.type == self.TYPE_SIMPLE:
                code = """
                import nltk
                nltk.download('punkt')
                nltk.download('averaged_perceptron_tagger')
                from nltk.tokenize import TweetTokenizer
                from nltk.tag import pos_tag
                from nltk.tokenize import word_tokenize

                {output} = {input}{copy_code}
                result = []
                toktok = TweetTokenizer()
        
                for row in {output}['{att}'].to_numpy():
                    result.append(pos_tag(word_tokenize(row))) 
                {output}['{alias}'] = result
                """.format(copy_code=copy_code, output=self.output,
                           input=self.named_inputs['input data'],
                           att=self.attributes[0], alias=self.alias[0],
                           limit=self.min_token_lenght)
            else:
                code = """
                nltk.download('wordnet')
                from nltk.tokenize import regexp_tokenize
                from nltk.tokenize import RegexpTokenizer
                from nltk.tag import pos_tag

                {output} = {input}{copy_code}
                result = []
    
                for row in {output}['{att}'].to_numpy():
                    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
                    result.append(pos_tag(tokenizer.tokenize(row)))
    
                {output}['{alias}'] = result
                """.format(copy_code=copy_code, output=self.output,
                          input=self.named_inputs['input data'],
                          att=self.attributes[0], alias=self.alias[0],
                          exp=self.expression_param, limit=self.min_token_lenght)

            return dedent(code)

class SynonymsOperation(Operation):
    """This operation generate synonyms of a word.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_synonym'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        import nltk
        nltk.download('wordnet')
        from nltk.corpus import wordnet
        def synonym(word):
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
    
            return synonyms

        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(synonym)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)
  
