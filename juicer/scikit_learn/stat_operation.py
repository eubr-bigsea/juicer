# -*- coding: utf-8 -*-
from textwrap import dedent

import re
from itertools import zip_longest
from juicer.operation import Operation


class PdfOperation(Operation):
    """Calculate the probability density function
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
        self.alias = [x[1] or '{}_pdf'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""

        code = f"""
        {self.output} = {self.input}
	
        alias = {self.alias}
        for i, attr in enumerate({self.attributes}):
	    tmp_sum = {self.input}[attr].sum()
            {self.output}[alias[i]] = {self.input}.apply(lambda x: x[[attr]]/tmp_sum, axis=1)
        """
        return dedent(code)

class CdfOperation(Operation):
    """Calculate the cumulative distribution function
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
        self.alias = [x[1] or '{}_cdf'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""

        code = f"""
        df = {self.input}
	
        alias = {self.alias}
        for i, attr in enumerate({self.attributes}):
	    tmp_sum = {self.input}[attr].sum()
            df[alias[i]] = {self.input}.apply(lambda x: x[[attr]]/tmp_sum, axis=1)
            for j in range(1, len(df[alias[i]])):
		df.loc[j, alias[i]] += df.loc[j-1, alias[i]]
	    {self.output} = df
        """
        return dedent(code)

class CcdfOperation(Operation):
    """Calculate the cumulative distribution function
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
        self.alias = [x[1] or '{}_cdf'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""

        code = f"""
        df = {self.input}
	
        alias = {self.alias}
        for i, attr in enumerate({self.attributes}):
	    tmp_sum = {self.input}[attr].sum()
            df[alias[i]] = {self.input}.apply(lambda x: x[[attr]]/tmp_sum, axis=1)
            for j in reversed(range(1, len(df[alias[i]]))):
		df.loc[j, alias[i]] += df.loc[j+1, alias[i]]
	    {self.output} = df
        """
        return dedent(code)

