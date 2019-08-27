# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


class LSTM(Operation):
    UNITS_PARAM = 'units'
    ACTIVATION_PARAM = 'activation'
    RECURRENT_ACTIVATION_PARAM = 'recurrent_activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    RECURRENT_INITIALIZER_PARAM = 'recurrent_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    UNIT_FORGET_BIAS_PARAM = 'unit_forget_bias'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    RECURRENT_REGULARIZER_PARAM = 'recurrent_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    RECURRENT_CONSTRAINT_PARAM = 'recurrent_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    DROPOUT_PARAM = 'dropout'
    RECURRENT_DROPOUT_PARAM = 'recurrent_dropout'
    IMPLEMENTATION_PARAM = 'implementation'
    RETURN_SEQUENCES_PARAM = 'return_sequences'
    RETURN_STATE_PARAM = 'return_state'
    GO_BACKWARDS_PARAM = 'go_backwards'
    STATEFUL_PARAM = 'stateful'
    UNROLL_PARAM = 'unroll'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.UNITS_PARAM not in parameters or self.UNITS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} are required').format(self.UNITS_PARAM))

        self.units = parameters.get(self.UNITS_PARAM)
        self.activation = parameters.get(self.ACTIVATION_PARAM, None) or None
        self.recurrent_activation = parameters.get(
            self.RECURRENT_ACTIVATION_PARAM, None) or None
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None) or None
        self.recurrent_initializer = parameters.get(
            self.RECURRENT_INITIALIZER_PARAM, None) or None
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None) or None
        self.unit_forget_bias = parameters.get(self.UNIT_FORGET_BIAS_PARAM)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None) or None
        self.recurrent_regularizer = parameters.get(
            self.RECURRENT_REGULARIZER_PARAM, None) or None
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None) or None
        self.activity_regularizer = parameters.get(
            self.ACTIVITY_REGULARIZER_PARAM, None) or None
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None) or None
        self.recurrent_constraint = parameters.get(
            self.RECURRENT_CONSTRAINT_PARAM, None) or None
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM,
                                              None) or None
        self.dropout = parameters.get(self.DROPOUT_PARAM)
        self.recurrent_dropout = parameters.get(self.RECURRENT_DROPOUT_PARAM)
        self.implementation = parameters.get(self.IMPLEMENTATION_PARAM)
        self.return_sequences = parameters.get(self.RETURN_SEQUENCES_PARAM)
        self.return_state = parameters.get(self.RETURN_STATE_PARAM)
        self.go_backwards = parameters.get(self.GO_BACKWARDS_PARAM)
        self.stateful = parameters.get(self.STATEFUL_PARAM)
        self.unroll = parameters.get(self.UNROLL_PARAM)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()

        if self.UNITS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.UNITS_PARAM))

        self.treatment()

        self.import_code = {'layer': 'LSTM',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def remove_python_code_parent(self):
        python_code_to_remove = []
        for parent in self.parents_by_port:
            if parent[0] == 'python code':
                python_code_to_remove.append(convert_parents_to_variable_name(
                    [parent[1]])
                )
        return python_code_to_remove

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parent.remove(python_code[0])

        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.units <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be a positive integer').format(
                    self.UNITS_PARAM))
        if self.dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.DROPOUT_PARAM))
        if self.recurrent_dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.RECURRENT_DROPOUT_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.unit_forget_bias = True if int(
            self.unit_forget_bias) == 1 else False
        self.return_sequences = True if int(
            self.return_sequences) == 1 else False
        self.return_state = True if int(self.return_state) == 1 else False
        self.go_backwards = True if int(self.go_backwards) == 1 else False
        self.stateful = True if int(self.stateful) == 1 else False
        self.unroll = True if int(self.unroll) == 1 else False
        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        self.implementation = int(self.implementation)

        functions_required = []
        self.units = """units='{units}'""".format(units=self.units)
        functions_required.append(self.units)

        if self.advanced_options:
            if self.activation is not None:
                self.activation = """activation='{activation}'""" \
                    .format(activation=self.activation)
                functions_required.append(self.activation)
            if self.recurrent_activation is not None:
                self.recurrent_activation = """recurrent_activation=
                '{recurrent_activation}'""".format(
                    recurrent_activation=self.recurrent_activation)
                functions_required.append(self.recurrent_activation)

            self.use_bias = """use_bias={use_bias}""".format(
                use_bias=self.use_bias)
            functions_required.append(self.use_bias)

            if self.kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer=
                '{kernel_initializer}'""".format(
                    kernel_initializer=self.kernel_initializer)
                functions_required.append(self.kernel_initializer)
            if self.recurrent_initializer is not None:
                self.recurrent_initializer = """recurrent_initializer=
                '{recurrent_initializer}'""".format(
                    recurrent_initializer=self.recurrent_initializer)
                functions_required.append(self.recurrent_initializer)
            if self.bias_initializer is not None:
                self.bias_initializer = """bias_initializer=
                '{bias_initializer}'""".format(
                    bias_initializer=self.bias_initializer)
                functions_required.append(self.bias_initializer)

            self.unit_forget_bias = """unit_forget_bias=
            {unit_forget_bias}""".format(unit_forget_bias=self.unit_forget_bias)
            functions_required.append(self.unit_forget_bias)

            if self.kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer=
                '{kernel_regularizer}'""".format(
                    kernel_regularizer=self.kernel_regularizer)
                functions_required.append(self.kernel_regularizer)
            if self.recurrent_regularizer is not None:
                self.recurrent_regularizer = """recurrent_regularizer=
                '{recurrent_regularizer}'""".format(
                    recurrent_regularizer=self.recurrent_regularizer)
                functions_required.append(self.recurrent_regularizer)
            if self.bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer=
                '{bias_regularizer}'""".format(
                    bias_regularizer=self.bias_regularizer)
                functions_required.append(self.bias_regularizer)
            if self.activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer=
                '{activity_regularizer}'""".format(
                    activity_regularizer=self.activity_regularizer)
                functions_required.append(self.activity_regularizer)
            if self.kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint=
                '{kernel_constraint}'""".format(
                    kernel_constraint=self.kernel_constraint)
                functions_required.append(self.kernel_constraint)
            if self.recurrent_constraint is not None:
                self.recurrent_constraint = """recurrent_constraint=
                '{recurrent_constraint}'""".format(
                    recurrent_constraint=self.recurrent_constraint)
                functions_required.append(self.recurrent_constraint)
            if self.bias_constraint is not None:
                self.bias_constraint = """bias_constraint=
                '{bias_constraint}'""".format(
                    bias_constraint=self.bias_constraint)
                functions_required.append(self.bias_constraint)

            self.dropout = """dropout={dropout}""".format(
                dropout=self.dropout)
            functions_required.append(self.dropout)

            self.recurrent_dropout = """recurrent_dropout=
            {recurrent_dropout}""".format(
                recurrent_dropout=self.recurrent_dropout)
            functions_required.append(self.recurrent_dropout)

            self.implementation = """implementation={implementation}""".format(
                implementation=self.implementation)
            functions_required.append(self.implementation)

            self.return_sequences = """return_sequences=
            {return_sequences}""".format(return_sequences=self.return_sequences)
            functions_required.append(self.return_sequences)

            self.return_state = """return_state={return_state}""".format(
                return_state=self.return_state)
            functions_required.append(self.return_state)

            self.go_backwards = """go_backwards={go_backwards}""".format(
                go_backwards=self.go_backwards)
            functions_required.append(self.go_backwards)

            self.stateful = """stateful={stateful}""".format(
                stateful=self.stateful)
            functions_required.append(self.stateful)

            self.unroll = """unroll={unroll}""".format(unroll=self.unroll)
            functions_required.append(self.unroll)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = LSTM(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class SimpleRNN(Operation):
    UNITS_PARAM = 'units'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    RECURRENT_INITIALIZER_PARAM = 'recurrent_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    RECURRENT_REGULARIZER_PARAM = 'recurrent_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    RECURRENT_CONSTRAINT_PARAM = 'recurrent_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    DROPOUT_PARAM = 'dropout'
    RECURRENT_DROPOUT_PARAM = 'recurrent_dropout'
    RETURN_SEQUENCES_PARAM = 'return_sequences'
    RETURN_STATE_PARAM = 'return_state'
    GO_BACKWARDS_PARAM = 'go_backwards'
    STATEFUL_PARAM = 'stateful'
    UNROLL_PARAM = 'unroll'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.UNITS_PARAM not in parameters or self.UNITS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} are required').format(self.UNITS_PARAM))

        self.units = parameters.get(self.UNITS_PARAM)
        self.activation = parameters.get(self.ACTIVATION_PARAM, None) or None
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None) or None
        self.recurrent_initializer = parameters.get(
            self.RECURRENT_INITIALIZER_PARAM, None) or None
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None) or None
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None) or None
        self.recurrent_regularizer = parameters.get(
            self.RECURRENT_REGULARIZER_PARAM, None) or None
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None) or None
        self.activity_regularizer = parameters.get(
            self.ACTIVITY_REGULARIZER_PARAM, None) or None
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None) or None
        self.recurrent_constraint = parameters.get(
            self.RECURRENT_CONSTRAINT_PARAM, None) or None
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM,
                                              None) or None
        self.dropout = parameters.get(self.DROPOUT_PARAM)
        self.recurrent_dropout = parameters.get(self.RECURRENT_DROPOUT_PARAM)
        self.return_sequences = parameters.get(self.RETURN_SEQUENCES_PARAM)
        self.return_state = parameters.get(self.RETURN_STATE_PARAM)
        self.go_backwards = parameters.get(self.GO_BACKWARDS_PARAM)
        self.stateful = parameters.get(self.STATEFUL_PARAM)
        self.unroll = parameters.get(self.UNROLL_PARAM)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()

        if self.UNITS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.UNITS_PARAM))

        self.treatment()

        self.import_code = {'layer': 'SimpleRNN',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def remove_python_code_parent(self):
        python_code_to_remove = []
        for parent in self.parents_by_port:
            if parent[0] == 'python code':
                python_code_to_remove.append(convert_parents_to_variable_name(
                    [parent[1]])
                )
        return python_code_to_remove

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parent.remove(python_code[0])

        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.units <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be a positive integer').format(
                    self.UNITS_PARAM))
        if self.dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.DROPOUT_PARAM))
        if self.recurrent_dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.RECURRENT_DROPOUT_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.return_sequences = True if int(
            self.return_sequences) == 1 else False
        self.return_state = True if int(self.return_state) == 1 else False
        self.go_backwards = True if int(self.go_backwards) == 1 else False
        self.stateful = True if int(self.stateful) == 1 else False
        self.unroll = True if int(self.unroll) == 1 else False
        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        self.implementation = int(self.implementation)

        functions_required = []
        self.units = """units='{units}'""".format(units=self.units)
        functions_required.append(self.units)

        if self.advanced_options:
            if self.activation is not None:
                self.activation = """activation='{activation}'""" \
                    .format(activation=self.activation)
                functions_required.append(self.activation)

            self.use_bias = """use_bias={use_bias}""".format(
                use_bias=self.use_bias)
            functions_required.append(self.use_bias)

            if self.kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer=
                '{kernel_initializer}'""".format(
                    kernel_initializer=self.kernel_initializer)
                functions_required.append(self.kernel_initializer)
            if self.recurrent_initializer is not None:
                self.recurrent_initializer = """recurrent_initializer=
                '{recurrent_initializer}'""".format(
                    recurrent_initializer=self.recurrent_initializer)
                functions_required.append(self.recurrent_initializer)
            if self.bias_initializer is not None:
                self.bias_initializer = """bias_initializer=
                '{bias_initializer}'""".format(
                    bias_initializer=self.bias_initializer)
                functions_required.append(self.bias_initializer)

            if self.kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer=
                '{kernel_regularizer}'""".format(
                    kernel_regularizer=self.kernel_regularizer)
                functions_required.append(self.kernel_regularizer)
            if self.recurrent_regularizer is not None:
                self.recurrent_regularizer = """recurrent_regularizer=
                '{recurrent_regularizer}'""".format(
                    recurrent_regularizer=self.recurrent_regularizer)
                functions_required.append(self.recurrent_regularizer)
            if self.bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer=
                '{bias_regularizer}'""".format(
                    bias_regularizer=self.bias_regularizer)
                functions_required.append(self.bias_regularizer)
            if self.activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer=
                '{activity_regularizer}'""".format(
                    activity_regularizer=self.activity_regularizer)
                functions_required.append(self.activity_regularizer)
            if self.kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint=
                '{kernel_constraint}'""".format(
                    kernel_constraint=self.kernel_constraint)
                functions_required.append(self.kernel_constraint)
            if self.recurrent_constraint is not None:
                self.recurrent_constraint = """recurrent_constraint=
                '{recurrent_constraint}'""".format(
                    recurrent_constraint=self.recurrent_constraint)
                functions_required.append(self.recurrent_constraint)
            if self.bias_constraint is not None:
                self.bias_constraint = """bias_constraint=
                '{bias_constraint}'""".format(
                    bias_constraint=self.bias_constraint)
                functions_required.append(self.bias_constraint)

            self.dropout = """dropout={dropout}""".format(
                dropout=self.dropout)
            functions_required.append(self.dropout)

            self.recurrent_dropout = """recurrent_dropout=
            {recurrent_dropout}""".format(
                recurrent_dropout=self.recurrent_dropout)
            functions_required.append(self.recurrent_dropout)

            self.return_sequences = """return_sequences=
            {return_sequences}""".format(return_sequences=self.return_sequences)
            functions_required.append(self.return_sequences)

            self.return_state = """return_state={return_state}""".format(
                return_state=self.return_state)
            functions_required.append(self.return_state)

            self.go_backwards = """go_backwards={go_backwards}""".format(
                go_backwards=self.go_backwards)
            functions_required.append(self.go_backwards)

            self.stateful = """stateful={stateful}""".format(
                stateful=self.stateful)
            functions_required.append(self.stateful)

            self.unroll = """unroll={unroll}""".format(unroll=self.unroll)
            functions_required.append(self.unroll)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = SimpleRNN(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)
