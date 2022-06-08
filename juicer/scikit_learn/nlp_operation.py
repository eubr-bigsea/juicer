# -*- coding: utf-8 -*-

from textwrap import dedent

import re
from itertools import zip_longest
from juicer.operation import Operation

class StemmingOperation(Operation):
    """Stemming is the process of producing morphological variants of a root/base word.
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
        # by filling missing alias with the attribute name suffixed by _stems.
        self.alias = [x[1] or '{}_stems'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 
    def generate_code(self):
        """Generate code."""

        code = """
        stemmer = SnowballStemmer("english")
        def stemming(word):
            return stemmer.stem(word)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(stemming)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class LemmatizationOperation(Operation):
    """Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item.
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
        self.alias = [x[1] or '{}_lemmas'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        def lemmatization(word):
            return lemmatizer.lemmatize(word)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(lemmatization)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)


class NormalizationOperation(Operation):
    """Normalization is the process of removing punctuaction and lower casing tokens.
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
        self.alias = [x[1] or '{}_normalized'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        def normalization(sentence):
            puncts='.?!'
            for punct in puncts:
                sentence = sentence.replace(punct, ' ')
            return sentence.lower()

        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(normalization)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class PosTaggingOperation(Operation):
    """Part-of-speech (POS) tagging is a process of labeling each word in a sentence with a morphosyntactic class
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
        self.alias = [x[1] or '{}_postags'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        nltk.download('averaged_perceptron_tagger')
        def posTagging(sentence):
            aux = eval(sentence)
            return nltk.pos_tag(aux)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(posTagging)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)
