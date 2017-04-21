# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class AddColumns(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)
        self.has_code = len(named_inputs) == 2
        self.has_import = "from functions.etl.etl_functions                import *"

    def generate_code(self):

        code = "{0} = AddColumns({1},{2})".format(self.named_outputs['output data'],
                                                  self.named_inputs['input data 1'],
                                                  self.named_inputs['input data 2'])
        return dedent(code)


class AddRows(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.parameters = parameters
        self.has_code = len(self.named_inputs) == 2 and len(self.named_outputs) > 0
        self.has_import = "from functions.etl.etl_functions                import *"

    def generate_code(self):

        code = "{0} = Union({1},{2})".format(self.named_outputs['output data'],
                                                self.named_inputs['input data 1'],
                                                self.named_inputs['input data 2'])
        return dedent(code)

class Distinct(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            self.attributes = []

        self.has_code = len(self.named_inputs) == 1
        self.has_import = "from functions.etl.etl_functions                import *"

    def generate_code(self):
        code = "{} = DropDuplicates({})".format(self.named_outputs['output data'],
                                                self.named_inputs['input data'])
        return dedent(code)


class Difference(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2
        self.has_import = "from functions.etl.etl_functions                import *"


    def generate_code(self):
        code = "{} = Difference({},{})".format( self.named_outputs['output data'],
                                              self.named_inputs['input data 1'],
                                              self.named_inputs['input data 2'])
        return dedent(code)


class Intersection(Operation):
    """
    Returns a new DataFrame containing rows only in both this frame and another frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.parameters = parameters
        self.has_code = len(self.named_inputs) == 2
        self.has_import = "from functions.etl.etl_functions                import *"

    def generate_code(self):

        code = "{} = Intersect({},{})".format(self.named_outputs['output data'],
                                              self.named_inputs['input data 1'],
                                              self.named_inputs['input data 2'])
        return dedent(code)


class Drop(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.column = parameters['column']
        self.has_import = "from functions.etl.etl_functions                import *"

    def generate_code(self):
        code = """{} = Drop({},'{}')""".format( self.outputs[0], self.inputs[0], self.column)
        return dedent(code)


class Filter(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.FILTER_PARAM not in parameters:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.FILTER_PARAM, self.__class__))

        self.filter = parameters.get(self.FILTER_PARAM)

        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        filters = ["(col('{0}') {1} '{2}')".format(f['attribute'], f['f'], f['value'])
            for f in self.filter]

        code = """{} = filter({},{})""".format(output, self.inputs[0], ' & '.join(filters))
        return dedent(code)


class Select(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_import = "from functions.etl.etl_functions                import *"
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(self.ATTRIBUTES_PARAM, self.__class__))


    def generate_code(self):

        code = """{} = Select({},[{}])""".format(self.named_outputs['output data'],
                                                 self.named_inputs['input data'],
                                                 ', '.join(['"{}"'.format(x) for x in self.attributes]))
        return dedent(code)
