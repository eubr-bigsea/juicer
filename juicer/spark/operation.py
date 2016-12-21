# -*- coding: utf-8 -*-
import logging
from textwrap import dedent

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Operation:
    """ Defines an operation in Lemonade """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        self.parameters = parameters
        self.inputs = inputs
        self.outputs = outputs
        self.named_inputs = named_inputs
        self.named_outputs = named_outputs
        self.multiple_inputs = False

        # Indicate if operation generates code or not. Some operations, e.g.
        # Comment, does not generate code
        self.has_code = len(self.inputs) > 0 or len(self.outputs) > 0

        # How many output ports the operation has
        self.expected_output_ports = 1

        if len(self.inputs) > 0:
            self.output = self.outputs[0] if len(
                self.outputs) else '{}_tmp_{}'.format(
                self.inputs[0], parameters['task']['order'])
        elif len(self.outputs) > 0:
            self.output = self.outputs[0]
        else:
            self.output = "NO_OUTPUT_WITHOUT_CONNECTIONS"

    def generate_code(self):
        raise NotImplementedError("Method generate_code should be implemented "
                                  "in {} subclass".format(self.__class__))

    @property
    def get_inputs_names(self):
        return ', '.join(self.inputs)

    def get_output_names(self, sep=", "):
        result = ''
        if len(self.outputs) > 0:
            result = sep.join(self.outputs)
        elif len(self.inputs) > 0:
            if self.expected_output_ports == 1:
                result = '{}_tmp_{}'.format(self.inputs[0],
                                            self.parameters['task']['order'])
        else:
            #raise ValueError(
            #    "Operation has neither input nor output: {}".format(
            #        self.__class__))
            pass
        return result

    def get_data_out_names(self, sep=','):
        return self.get_output_names(sep)

    def test_null_operation(self):
        """
        Test if an operation is null, i.e, does nothing.
        An operation does nothing if it has zero inputs or outputs.
        """
        return any([len(self.outputs) == 0, len(self.inputs) == 0])


class ReportOperation(Operation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

    def generate_code(self):
        raise NotImplementedError("Method generate_code should be implemented "
                                  "in {} subclass".format(self.__class__))


class DatetimeToBins(Operation):
    """
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.target_column = parameters['target_column']
        self.new_column = parameters['new_column']
        self.group_size = parameters['group_size']

    def generate_code(self):
        code = '''
            from bins import *
            {} = datetime_to_bins({}, {}, '{}', '{}')
            '''.format(self.outputs[0], self.inputs[0], self.group_size,
                       self.target_column, self.new_column)
        return dedent(code)


class NoOp(Operation):
    """ Null operation """

    def generate_code(self):
        pass

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.parameters = parameters
        self.has_code = False
