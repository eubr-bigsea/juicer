# -*- coding: utf-8 -*-
import ast
import pprint
import numpy as np
from textwrap import dedent
from juicer.operation import Operation
from itertools import zip_longest


class WordCountingOperation(Operation):
    """ Counts the words
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))


        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 
        self.transpiler_utils.add_import("from collections import Counter")
        self.transpiler_utils.add_import("from nltk.tokenize import RegexpTokenizer")
        self.transpiler_utils.add_import("import nltk")


    def generate_code(self):
        """Generate code."""

        code = f"""
        tokenizer = RegexpTokenizer(r'\w+', flags=re.IGNORECASE)

        df = pd.DataFrame()
        
        for i, attr in enumerate({self.attributes}):
            for j, row in {self.input}.iterrows():
                item = dict(row.iteritems())
                key = list(item.keys())[0]

                text = list(item.values())[0]
                tokens = tokenizer.tokenize(text)
                counter = Counter(tokens)
                words = dict(counter)
                to_append = dict()
                to_append[attr] = words

                df = df.append(to_append, ignore_index=True)
                
            df = df.fillna(0)
	
        {self.output} = df
        """
        return dedent(code)


class LowerCaseOperation(Operation):
    """ Lowers the case of words
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
        self.alias = [x[1] or '{}_lowercase'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""

        code = f"""

        df = pd.DataFrame()
        
        alias = {self.alias}
        for i, attr in enumerate({self.attributes}):
            for j, row in {self.input}.iterrows():
                item = dict(row.iteritems())
                key = list(item.keys())[0]

                text = list(item.values())[0]

                to_append = dict()
                to_append[alias[i]] = text.lower()

                df = df.append(to_append, ignore_index=True)
                
            df = df.fillna(0)
	
        {self.output} = df
        """
        return dedent(code)

