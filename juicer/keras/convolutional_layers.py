# -*- coding: utf-8 -*-
from gettext import gettext
from textwrap import dedent
from juicer.operation import Operation
import re
from ast import parse
from juicer.util.template_util import *


class Conv2DTranspose(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    OUTPUT_PADDING_PARAM = 'output_padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = int(parameters.get(self.FILTERS_PARAM))
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._output_padding = parameters.get(self.OUTPUT_PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None)
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                  None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                  None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(
            self.ACTIVITY_REGULARIZER_PARAM, None)
        self._kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                 None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM,
                                               None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.output_padding = None
        self.data_format = None
        self.dilation_rate = None
        self.activation = None
        self.use_bias = None
        self.kernel_initializer = None
        self.bias_initializer = None
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.kernel_constraint = None
        self.bias_constraint = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Conv2DTranspose',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.kernel_size = get_int_or_tuple(self._kernel_size)
        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.KERNEL_SIZE_PARAM))

        self.strides = get_int_or_tuple(self._strides)
        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.STRIDES_PARAM))

        self.dilation_rate = get_int_or_tuple(self._dilation_rate)
        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.DILATION_RATE_PARAM))

        self.output_padding = get_int_or_tuple(self._output_padding)
        if self.output_padding is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.OUTPUT_PADDING_PARAM))

        functions_required = []
        self.filters = """filters={filters}""".format(filters=self._filters)
        functions_required.append(self.filters)

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            functions_required.append("""kernel_size={kernel_size}"""
                                      .format(kernel_size=self.kernel_size))

            functions_required.append("""strides={strides}"""
                                      .format(strides=self.strides))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is not False:
                    functions_required.append("""input_shape='{input_shape}'"""
                                              .format(input_shape=self
                                                      .input_shape))
                else:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))

            if self._padding is not None:
                self.padding = """padding='{padding}'"""\
                    .format(padding=self._padding)
                functions_required.append(self.padding)

            if self._output_padding is not None:
                self.output_padding = """output_padding={output_padding}""" \
                    .format(output_padding=self._output_padding)
                functions_required.append(self.output_padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""" \
                    .format(data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._dilation_rate is not None:
                self.dilation_rate = """dilation_rate={dilation_rate}""" \
                    .format(dilation_rate=self._dilation_rate)
                functions_required.append(self.dilation_rate)

            if self._activation is not None:
                self.activation = """activation='{activation}'""" \
                    .format(activation=self._activation)
                functions_required.append(self.activation)

            self.use_bias = True if int(self._use_bias) == 1 else False
            functions_required.append("""use_bias={use_bias}"""
                                      .format(use_bias=self.use_bias))

            if self._kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer='{k_init}'""" \
                    .format(k_init=self._kernel_initializer)
                functions_required.append(self.kernel_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{bias_init}'""" \
                    .format(bias_init=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer='{k_reg}'""" \
                    .format(k_reg=self._kernel_regularizer)
                functions_required.append(self.kernel_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{bias_regu}'""" \
                    .format(bias_regu=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer='{a_rg}'"""\
                    .format(a_rg=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint='{k_const}'""" \
                    .format(k_const=self._kernel_constraint)
                functions_required.append(self.kernel_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b_constraint}'""" \
                    .format(b_constraint=self._bias_constraint)
                functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv2DTranspose(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Conv3DTranspose(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    OUTPUT_PADDING_PARAM = 'output_padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = int(parameters.get(self.FILTERS_PARAM))
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._output_padding = parameters.get(self.OUTPUT_PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None)
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                  None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                  None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(self.
                                                    ACTIVITY_REGULARIZER_PARAM,
                                                    None)
        self._kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                 None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.output_padding = None
        self.data_format = None
        self.dilation_rate = None
        self.activation = None
        self.use_bias = None
        self.kernel_initializer = None
        self.bias_initializer = None
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.kernel_constraint = None
        self.bias_constraint = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Conv3DTranspose',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []

        if self.filters > 0:
            functions_required.append("""filters={filters}""".format(
                filters=self.filters))
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.kernel_size = get_int_or_tuple(self._kernel_size)
        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.KERNEL_SIZE_PARAM))

        self.strides = get_int_or_tuple(self._strides)
        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.STRIDES_PARAM))

        self.dilation_rate = get_int_or_tuple(self._dilation_rate)
        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.DILATION_RATE_PARAM))

        self.output_padding = get_int_or_tuple(self._output_padding)
        if self.output_padding is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.OUTPUT_PADDING_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            functions_required.append("""kernel_size={kernel_size}""".format(
                kernel_size=self.kernel_size))

            functions_required.append("""strides={strides}""".format(
                strides=self.strides))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                functions_required.append(
                    """input_shape='{input_shape}'""".format(
                        input_shape=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""" \
                    .format(padding=self._padding)
                functions_required.append(self.padding)

            if self._output_padding is not None:
                self.output_padding = """output_padding={output_padding}""" \
                    .format(output_padding=self._output_padding)
                functions_required.append(self.output_padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""" \
                    .format(data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._dilation_rate is not None:
                self.dilation_rate = """dilation_rate={dilation_rate}""" \
                    .format(dilation_rate=self._dilation_rate)
                functions_required.append(self.dilation_rate)

            if self._activation is not None:
                self.activation = """activation='{activation}'""" \
                    .format(activation=self._activation)
                functions_required.append(self.activation)

            self.use_bias = True if int(self._use_bias) == 1 else False
            functions_required.append("""use_bias={use_bias}""".format(
                use_bias=self.use_bias))

            if self._kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer='{k_init}'""" \
                    .format(k_init=self._kernel_initializer)
                functions_required.append(self.kernel_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{b_init}'""" \
                    .format(b_init=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer='{k_reg}'""" \
                    .format(k_reg=self._kernel_regularizer)
                functions_required.append(self.kernel_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{b_reg}'""" \
                    .format(b_reg=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer='{a_r}'""" \
                    .format(a_r=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint='{k_const}'""" \
                    .format(k_const=self._kernel_constraint)
                functions_required.append(self.kernel_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b_constraint}'""" \
                    .format(b_constraint=self._bias_constraint)
                functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv3DTranspose(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Convolution1D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    TRAINABLE_PARAM = 'trainable'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = parameters.get(self.FILTERS_PARAM)
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None)
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._trainable = parameters.get(self.TRAINABLE_PARAM)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                  None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                  None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(self.
                                                    ACTIVITY_REGULARIZER_PARAM,
                                                    None)
        self._kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                 None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.data_format = None
        self.dilation_rate = None
        self.activation = None
        self.trainable = None
        self.use_bias = None
        self.kernel_initializer = None
        self.bias_initializer = None
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.kernel_constraint = None
        self.bias_constraint = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Conv1D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        self.filters = int(self._filters)
        if self.filters > 0:
            functions_required.append("""filters={filters}""".format(
                filters=self.filters))
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.strides = get_int_or_tuple(self._strides)
        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.STRIDES_PARAM))

        self.dilation_rate = get_int_or_tuple(self._dilation_rate)
        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.DILATION_RATE_PARAM))

        self.use_bias = True if int(self._use_bias) == 1 else False
        self.trainable = True if int(self._trainable) == 1 else False
        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False

        if self.advanced_options:
            self.kernel_size = get_int_or_tuple(self._kernel_size)
            if self.kernel_size is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.KERNEL_SIZE_PARAM))
            else:
                functions_required.append("""kernel_size={}""".format(
                    self._kernel_size))

            self.strides = """strides={strides}""".format(strides=self._strides)
            functions_required.append(self.strides)

            self.use_bias = """use_bias={use_bias}""".format(
                use_bias=self._use_bias)
            functions_required.append(self.use_bias)

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                functions_required.append(
                    """input_shape='{input_shape}'""".format(
                        input_shape=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._dilation_rate is not None:
                self.dilation_rate = """dilation_rate={dilation_rate}""".format(
                    dilation_rate=self._dilation_rate)
                functions_required.append(self.dilation_rate)

            if self._activation is not None:
                self.activation = """activation='{activation}'""" \
                    .format(activation=self._activation)
                functions_required.append(self.activation)

            if self._kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer=
                '{kernel_initializer}'""" \
                    .format(kernel_initializer=self._kernel_initializer)
                functions_required.append(self.kernel_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{bias_init}'""" \
                    .format(bias_init=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer='{k_reg}'""" \
                    .format(k_reg=self._kernel_regularizer)
                functions_required.append(self.kernel_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{bias_reg}'""" \
                    .format(bias_reg=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer='{a_r}'""" \
                    .format(a_r=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint='{k_const'""" \
                    .format(k_const=self._kernel_constraint)
                functions_required.append(self.kernel_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{bias_const}'""" \
                    .format(bias_const=self._bias_constraint)
                functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv1D(
                name='{name}'{add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class Convolution2D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    TRAINABLE_PARAM = 'trainable'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    WEIGHTS_PARAM = 'weights'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = int(parameters.get(self.FILTERS_PARAM))
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None)
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._trainable = parameters.get(self.TRAINABLE_PARAM, 0)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM, 0)
        self._kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                  None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                  None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(
            self.ACTIVITY_REGULARIZER_PARAM,
            None)
        self._kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                 None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
        self._weights = parameters.get(self.WEIGHTS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.data_format = None
        self.dilation_rate = None
        self.activation = None
        self.trainable = None
        self.use_bias = None
        self.kernel_initializer = None
        self.bias_initializer = None
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.kernel_constraint = None
        self.bias_constraint = None
        self.weights = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Conv2D',
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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        self.filters = int(self._filters)
        if self.filters > 0:
            functions_required.append("""filters={filters}""".format(
                filters=self.filters))
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.kernel_size = get_int_or_tuple(self._kernel_size)
        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.KERNEL_SIZE_PARAM))
        else:
            functions_required.append("""kernel_size={kernel_size}""".format(
                kernel_size=self.kernel_size))

        self.use_bias = True if int(self._use_bias) == 1 else False
        self.trainable = True if int(self._trainable) == 1 else False
        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False

        if self.advanced_options:
            self.strides = get_int_or_tuple(self._strides)
            if self.strides is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.STRIDES_PARAM))
            else:
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            functions_required.append("""use_bias={use_bias}""".format(
                use_bias=self.use_bias))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                else:
                    functions_required.append(
                        """input_shape='{input_shape}'""".format(
                            input_shape=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            self.dilation_rate = get_int_or_tuple(self._dilation_rate)
            if self.dilation_rate is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.DILATION_RATE_PARAM))
            if self.dilation_rate is not None:
                functions_required.append(
                    """dilation_rate={dilation_rate}""".format(
                        dilation_rate=self.dilation_rate))

            if self._activation is not None:
                self.activation = """activation='{activation}'""".format(
                    activation=self._activation)
                functions_required.append(self.activation)

            if self._kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer='{k}'""".format(
                    k=self._kernel_initializer)
                functions_required.append(self.kernel_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{b}'""".format(
                    b=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer='{k}'""".format(
                    kernel_regularizer=self._kernel_regularizer)
                functions_required.append(self.kernel_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{b}'""".format(
                    b=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer='{a}'""".format(
                    a=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint='{k}'""".format(
                    k=self._kernel_constraint)
                functions_required.append(self.kernel_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b}'""".format(
                    b=self._bias_constraint)
                functions_required.append(self.bias_constraint)

            if self._weights is not None and self._weights.strip():
                if convert_to_list(self._weights):
                    self.weights = """weights={weights}""".format(
                        weights=self._weights)
                    functions_required.append(self.weights)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv2D(
                name='{name}'{add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class Convolution3D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    TRAINABLE_PARAM = 'trainable'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    WEIGHTS_PARAM = 'weights'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = parameters.get(self.FILTERS_PARAM)
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None)
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._trainable = parameters.get(self.TRAINABLE_PARAM)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                  None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                  None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(self.
                                                    ACTIVITY_REGULARIZER_PARAM,
                                                    None)
        self._kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                 None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
        self._weights = parameters.get(self.WEIGHTS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.data_format = None
        self.dilation_rate = None
        self.activation = None
        self.trainable = None
        self.use_bias = None
        self.kernel_initializer = None
        self.bias_initializer = None
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.kernel_constraint = None
        self.bias_constraint = None
        self.weights = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Conv3D',
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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        self.filters = int(self._filters)
        if self.filters > 0:
            functions_required.append("""filters={filters}""".format(
                filters=self.filters))
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self.strides is not None:
                self.strides = get_int_or_tuple(self._strides)
                if self.strides is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.STRIDES_PARAM))
                else:
                    functions_required.append("""strides={strides}""".format(
                        strides=self.strides))

            if self._dilation_rate is not None:
                self.dilation_rate = get_int_or_tuple(self._dilation_rate)
                if self.dilation_rate is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.DILATION_RATE_PARAM))
                else:
                    functions_required.append(
                        """dilation_rate={dilation_rate}""".format(
                            dilation_rate=self.dilation_rate))

            if self._kernel_size is not None:
                self.kernel_size = get_int_or_tuple(self._kernel_size)
                if self.kernel_size is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.KERNEL_SIZE_PARAM))
                else:
                    functions_required.append(
                        """kernel_size={kernel_size}""".format(
                            kernel_size=self.kernel_size))

            self.trainable = True if int(self._trainable) == 1 else False
            self.use_bias = True if int(self._use_bias) == 1 else False
            functions_required.append("""use_bias={use_bias}""".format(
                use_bias=self.use_bias))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                else:
                    functions_required.append(
                        """input_shape='{input_shape}'""".format(
                            input_shape=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""" \
                    .format(padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""" \
                    .format(data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._activation is not None:
                self.activation = """activation='{activation}'""" \
                    .format(activation=self._activation)
                functions_required.append(self.activation)

            if self._kernel_initializer is not None:
                self.kernel_initializer = """kernel_initializer='{k}'""".format(
                    k=self._kernel_initializer)
                functions_required.append(self.kernel_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{b}'""".format(
                    b=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._kernel_regularizer is not None:
                self.kernel_regularizer = """kernel_regularizer='{k}'""".format(
                    k=self._kernel_regularizer)
                functions_required.append(self.kernel_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{b}'""".format(
                    b=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = """activity_regularizer='{a}'""".format(
                    a=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._kernel_constraint is not None:
                self.kernel_constraint = """kernel_constraint='{k}'""".format(
                    k=self._kernel_constraint)
                functions_required.append(self.kernel_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b}'""".format(
                    b=self._bias_constraint)
                functions_required.append(self.bias_constraint)

            if self._weights is not None and self._weights.strip():
                if convert_to_list(self._weights):
                    self.weights = """weights={weights}""".format(
                        weights=self._weights)
                    functions_required.append(self.weights)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv3D(
                name='{name}'{add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class Cropping1D(Operation):
    CROPPING_PARAM = 'cropping'
    INPUT_SHAPE_PARAM = 'input_shape'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.CROPPING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required.').format(
                self.CROPPING_PARAM)
            )

        self._cropping = abs(parameters.get(self.CROPPING_PARAM))
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.cropping = None
        self.input_shape = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Cropping1D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []

        if self._cropping.strip():
            self.cropping = tuple_of_tuples(self._cropping)
            if self.cropping is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.CROPPING_PARAM))
            else:
                functions_required.append("""cropping={cropping}""".format(
                    cropping=self.cropping))
        else:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.CROPPING_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                else:
                    functions_required.append(
                        """input_shape={input_shape}""".format(
                            input_shape=self.input_shape))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Cropping1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Cropping2D(Operation):
    CROPPING_PARAM = 'cropping'
    INPUT_SHAPE_PARAM = 'input_shape'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.CROPPING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required.').format(
                self.CROPPING_PARAM)
            )

        self._cropping = abs(parameters.get(self.CROPPING_PARAM))
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.cropping = None
        self.input_shape = None
        self.data_format = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Cropping2D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []

        if self._cropping.strip():
            self.cropping = tuple_of_tuples(self._cropping)
            if self.cropping is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.CROPPING_PARAM))
            else:
                functions_required.append("""cropping={cropping}""".format(
                    cropping=self.cropping))
        else:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.CROPPING_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                else:
                    functions_required.append(
                        """input_shape={input_shape}""".format(
                            input_shape=self.input_shape))

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Cropping2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Cropping3D(Operation):
    CROPPING_PARAM = 'cropping'
    INPUT_SHAPE_PARAM = 'input_shape'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.CROPPING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required.').format(
                self.CROPPING_PARAM)
            )

        self._cropping = abs(parameters.get(self.CROPPING_PARAM))
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.cropping = None
        self.input_shape = None
        self.data_format = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Cropping3D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []

        if self._cropping.strip():
            self.cropping = tuple_of_tuples(self._cropping)
            if self.cropping is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.CROPPING_PARAM))
            else:
                functions_required.append("""cropping={cropping}""".format(
                    cropping=self.cropping))
        else:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.CROPPING_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                else:
                    functions_required.append(
                        """input_shape={input_shape}""".format(
                            input_shape=self.input_shape))

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Cropping3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class DepthwiseConv2D(Operation):
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DEPTH_MULTIPLIER_PARAM = 'depth_multiplier'
    DATA_FORMAT_PARAM = 'data_format'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    DEPTHWISE_INITIALIZER_PARAM = 'depthwise_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    DEPTHWISE_REGULARIZER_PARAM = 'depthwise_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    DEPTHWISE_CONSTRAINT_PARAM = 'depthwise_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.KERNEL_SIZE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required.').format(
                self.KERNEL_SIZE_PARAM)
            )

        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._depth_multiplier = abs(parameters.get(self.DEPTH_MULTIPLIER_PARAM,
                                                    None))
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._depthwise_initializer = parameters.get(self.
                                                     DEPTHWISE_INITIALIZER_PARAM,
                                                     None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._depthwise_regularizer = parameters.get(self.
                                                     DEPTHWISE_REGULARIZER_PARAM,
                                                     None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(self.
                                                    ACTIVITY_REGULARIZER_PARAM,
                                                    None)
        self._depthwise_constraint = parameters.get(self.
                                                    DEPTHWISE_CONSTRAINT_PARAM,
                                                    None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.data_format = None
        self.depth_multiplier = None
        self.activation = None
        self.use_bias = None
        self.depthwise_initializer = None
        self.bias_initializer = None
        self.depthwise_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.depthwise_constraint = None
        self.bias_constraint = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'DepthwiseConv2D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        if self.KERNEL_SIZE_PARAM not in self.parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.KERNEL_SIZE_PARAM)
            )

        functions_required = []
        self.kernel_size = get_int_or_tuple(self._kernel_size)
        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.KERNEL_SIZE_PARAM))
        else:
            functions_required.append("""kernel_size={kernel_size}""".format(
                kernel_size=self.kernel_size))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._strides is not None:
                self.strides = get_int_or_tuple(self._strides)
                if self.strides is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.STRIDES_PARAM))
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                functions_required.append("""input_shape='{input}'""".format(
                    input=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._depth_multiplier is not None:
                self.depth_multiplier = """depth_multiplier={dm}""".format(
                    dm=self._depth_multiplier)
                functions_required.append(self.depth_multiplier)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._activation is not None:
                self.activation = """activation='{activation}'""".format(
                    activation=self._activation)
                functions_required.append(self.activation)

            self.use_bias = True if int(self._use_bias) == 1 else False
            functions_required.append("""use_bias={use_bias}""".format(
                use_bias=self.use_bias))

            if self._depthwise_initializer is not None:
                self.depthwise_initializer = \
                """depthwise_initializer='{depthwise_initializer}'""".format(
                    depthwise_initializer=self._depthwise_initializer)
                functions_required.append(self.depthwise_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{b}'""".format(
                    b=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._depthwise_regularizer is not None:
                self.depthwise_regularizer = \
                    """depthwise_regularizer='{d}'""".format(
                        d=self._depthwise_regularizer)
                functions_required.append(self.depthwise_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{b}'""".format(
                    b=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = \
                    """activity_regularizer='{activity_regularizer}'""".format(
                        activity_regularizer=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._depthwise_constraint is not None:
                self.depthwise_constraint = \
                    """depthwise_constraint='{depthwise_constraint}'""".format(
                        depthwise_constraint=self._depthwise_constraint)
                functions_required.append(self.depthwise_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b}'""".format(
                    b=self._bias_constraint)
                functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = DepthwiseConv2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class SeparableConv1D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    DEPTH_MULTIPLIER_PARAM = 'depth_multiplier'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    DEPTHWISE_INITIALIZER_PARAM = 'depthwise_initializer'
    POINTWISE_INITIALIZER_PARAM = 'pointwise_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    DEPTHWISE_REGULARIZER_PARAM = 'depthwise_regularizer'
    POINTWISE_REGULARIZER_PARAM = 'pointwise_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    DEPTHWISE_CONSTRAINT_PARAM = 'depthwise_constraint'
    POINTWISE_CONSTRAINT_PARAM = 'pointwise_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = abs(parameters.get(self.FILTERS_PARAM))
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                              None
        self._depth_multiplier = abs(parameters.get(self.DEPTH_MULTIPLIER_PARAM,
                                                    None))
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._depthwise_initializer = parameters.get(self.
                                                     DEPTHWISE_INITIALIZER_PARAM,
                                                     None)
        self._pointwise_initializer = parameters.get(self.
                                                     POINTWISE_INITIALIZER_PARAM,
                                                     None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._depthwise_regularizer = parameters.get(self.
                                                     DEPTHWISE_REGULARIZER_PARAM,
                                                     None)
        self._pointwise_regularizer = parameters.get(self.
                                                     POINTWISE_REGULARIZER_PARAM,
                                                     None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(self.
                                                    ACTIVITY_REGULARIZER_PARAM,
                                                    None)
        self._depthwise_constraint = parameters.get(self.
                                                    DEPTHWISE_CONSTRAINT_PARAM,
                                                    None)
        self._pointwise_constraint = parameters.get(self.
                                                    POINTWISE_CONSTRAINT_PARAM,
                                                    None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
                              
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.data_format = None
        self.dilation_rate = None
        self.depth_multiplier = None
        self.activation = None
        self.use_bias = None
        self.depthwise_initializer = None
        self.pointwise_initializer = None
        self.bias_initializer = None
        self.depthwise_regularizer = None
        self.pointwise_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.depthwise_constraint = None
        self.pointwise_constraint = None
        self.bias_constraint = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'SeparableConv1D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        if self._filters > 0:
            self.filters = """filters={filters}""".format(filters=self._filters)
            functions_required.append(self.filters)
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kernel_size is not None:
                self.kernel_size = get_int_or_tuple(self._kernel_size)
                if self.kernel_size is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.KERNEL_SIZE_PARAM))
                functions_required.append(
                    """kernel_size={kernel_size}""".format(
                        kernel_size=self.kernel_size))

            if self._strides is not None:
                self.strides = get_int_or_tuple(self._strides)
                if self.strides is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.STRIDES_PARAM))
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._dilation_rate is not None:
                self.dilation_rate = get_int_or_tuple(self._dilation_rate)
                if self.dilation_rate is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.DILATION_RATE_PARAM))
                functions_required.append(
                    """dilation_rate={dilation_rate}""".format(
                        dilation_rate=self.dilation_rate))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                functions_required.append(
                    """input_shape='{input_shape}'""".format(
                        input_shape=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._depth_multiplier is not None:
                self.depth_multiplier = """depth_multiplier={dm}""".format(
                    dm=self._depth_multiplier)
                functions_required.append(self.depth_multiplier)

            if self._activation is not None:
                self.activation = """activation='{activation}'""".format(
                    activation=self._activation)
                functions_required.append(self.activation)

            self.use_bias = True if int(self._use_bias) == 1 else False
            functions_required.append("""use_bias={use_bias}""".format(
                use_bias=self.use_bias))

            if self._depthwise_initializer is not None:
                self.depthwise_initializer = \
                    """depthwise_initializer='{depthwise_init}'""".format(
                        depthwise_init=self._depthwise_initializer)
                functions_required.append(self.depthwise_initializer)

            if self._pointwise_initializer is not None:
                self.pointwise_initializer = \
                    """pointwise_initializer='{pointwise_init}'""".format(
                        pointwise_init=self._pointwise_initializer)
                functions_required.append(self.pointwise_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{b}'""".format(
                    b=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._depthwise_regularizer is not None:
                self.depthwise_regularizer = \
                    """depthwise_regularizer='{depthwise_reg}'""".format(
                        depthwise_reg=self._depthwise_regularizer)
                functions_required.append(self.depthwise_regularizer)

            if self._pointwise_regularizer is not None:
                self.pointwise_regularizer = \
                    """pointwise_regularizer='{pointwise_reg}'""".format(
                        pointwise_reg=self._pointwise_regularizer)
                functions_required.append(self.pointwise_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{b_reg}'""".format(
                    b_reg=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = \
                    """activity_regularizer='{activity_regularizer}'""".format(
                        activity_regularizer=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._depthwise_constraint is not None:
                self.depthwise_constraint = \
                    """depthwise_constraint='{depthwise_constraint}'""".format(
                        depthwise_constraint=self._depthwise_constraint)
                functions_required.append(self.depthwise_constraint)

            if self._pointwise_constraint is not None:
                self.pointwise_constraint = \
                    """pointwise_constraint='{pointwise_constraint}'""".format(
                        pointwise_constraint=self._pointwise_constraint)
                functions_required.append(self.pointwise_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b_const}'""".format(
                    b_const=self._bias_constraint)
                functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = SeparableConv1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class SeparableConv2D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    DEPTH_MULTIPLIER_PARAM = 'depth_multiplier'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    DEPTHWISE_INITIALIZER_PARAM = 'depthwise_initializer'
    POINTWISE_INITIALIZER_PARAM = 'pointwise_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    DEPTHWISE_REGULARIZER_PARAM = 'depthwise_regularizer'
    POINTWISE_REGULARIZER_PARAM = 'pointwise_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    DEPTHWISE_CONSTRAINT_PARAM = 'depthwise_constraint'
    POINTWISE_CONSTRAINT_PARAM = 'pointwise_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required.').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self._filters = abs(parameters.get(self.FILTERS_PARAM))
        self._kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self._strides = parameters.get(self.STRIDES_PARAM)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                              None
        self._depth_multiplier = abs(parameters.get(self.DEPTH_MULTIPLIER_PARAM,
                                                    None))
        self._activation = parameters.get(self.ACTIVATION_PARAM, None)
        self._use_bias = parameters.get(self.USE_BIAS_PARAM)
        self._depthwise_initializer = parameters.get(self.
                                                     DEPTHWISE_INITIALIZER_PARAM,
                                                     None)
        self._pointwise_initializer = parameters.get(self.
                                                     POINTWISE_INITIALIZER_PARAM,
                                                     None)
        self._bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                                None)
        self._depthwise_regularizer = parameters.get(self.
                                                     DEPTHWISE_REGULARIZER_PARAM,
                                                     None)
        self._pointwise_regularizer = parameters.get(self.
                                                     POINTWISE_REGULARIZER_PARAM,
                                                     None)
        self._bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                                None)
        self._activity_regularizer = parameters.get(self.
                                                    ACTIVITY_REGULARIZER_PARAM,
                                                    None)
        self._depthwise_constraint = parameters.get(self.
                                                    DEPTHWISE_CONSTRAINT_PARAM,
                                                    None)
        self._pointwise_constraint = parameters.get(self.
                                                    POINTWISE_CONSTRAINT_PARAM,
                                                    None)
        self._bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)

        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.filters = None
        self.kernel_size = None
        self.strides = None
        self.input_shape = None
        self.padding = None
        self.data_format = None
        self.dilation_rate = None
        self.depth_multiplier = None
        self.activation = None
        self.use_bias = None
        self.depthwise_initializer = None
        self.pointwise_initializer = None
        self.bias_initializer = None
        self.depthwise_regularizer = None
        self.pointwise_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.depthwise_constraint = None
        self.pointwise_constraint = None
        self.bias_constraint = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'SeparableConv2D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        if self._filters > 0:
            self.filters = """filters={filters}""".format(filters=self._filters)
            functions_required.append(self.filters)
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.FILTERS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kernel_size is not None:
                self.kernel_size = get_int_or_tuple(self._kernel_size)
                if self.kernel_size is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.KERNEL_SIZE_PARAM))
                functions_required.append(
                    """kernel_size={kernel_size}""".format(
                        kernel_size=self.kernel_size))

            if self._strides is not None:
                self.strides = get_int_or_tuple(self._strides)
                if self.strides is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.STRIDES_PARAM))
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._dilation_rate is not None:
                self.dilation_rate = get_int_or_tuple(self._dilation_rate)
                if self.dilation_rate is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.DILATION_RATE_PARAM))
                functions_required.append(
                    """dilation_rate={dilation_rate}""".format(
                        dilation_rate=self.dilation_rate))

            if self._input_shape is not None:
                self.input_shape = get_int_or_tuple(self._input_shape)
                if self.input_shape is False:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.INPUT_SHAPE_PARAM))
                functions_required.append(
                    """input_shape='{input_shape}'""".format(
                        input_shape=self.input_shape))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._depth_multiplier is not None:
                self.depth_multiplier = """depth_multiplier={dm}""".format(
                    dm=self._depth_multiplier)
                functions_required.append(self.depth_multiplier)

            if self._activation is not None:
                self.activation = """activation='{activation}'""".format(
                    activation=self._activation)
                functions_required.append(self.activation)

            self.use_bias = True if int(self._use_bias) == 1 else False
            functions_required.append("""use_bias={use_bias}""".format(
                use_bias=self.use_bias))

            if self._depthwise_initializer is not None:
                self.depthwise_initializer = \
                    """depthwise_initializer='{depthwise_init}'""".format(
                        depthwise_init=self._depthwise_initializer)
                functions_required.append(self.depthwise_initializer)

            if self._pointwise_initializer is not None:
                self.pointwise_initializer = \
                    """pointwise_initializer='{pointwise_init}'""".format(
                        pointwise_init=self._pointwise_initializer)
                functions_required.append(self.pointwise_initializer)

            if self._bias_initializer is not None:
                self.bias_initializer = """bias_initializer='{b}'""".format(
                    b=self._bias_initializer)
                functions_required.append(self.bias_initializer)

            if self._depthwise_regularizer is not None:
                self.depthwise_regularizer = \
                    """depthwise_regularizer='{depthwise_reg}'""".format(
                        depthwise_reg=self._depthwise_regularizer)
                functions_required.append(self.depthwise_regularizer)

            if self._pointwise_regularizer is not None:
                self.pointwise_regularizer = \
                    """pointwise_regularizer='{pointwise_reg}'""".format(
                        pointwise_reg=self._pointwise_regularizer)
                functions_required.append(self.pointwise_regularizer)

            if self._bias_regularizer is not None:
                self.bias_regularizer = """bias_regularizer='{b_reg}'""".format(
                    b_reg=self._bias_regularizer)
                functions_required.append(self.bias_regularizer)

            if self._activity_regularizer is not None:
                self.activity_regularizer = \
                    """activity_regularizer='{activity_regularizer}'""".format(
                        activity_regularizer=self._activity_regularizer)
                functions_required.append(self.activity_regularizer)

            if self._depthwise_constraint is not None:
                self.depthwise_constraint = \
                    """depthwise_constraint='{depthwise_constraint}'""".format(
                        depthwise_constraint=self._depthwise_constraint)
                functions_required.append(self.depthwise_constraint)

            if self._pointwise_constraint is not None:
                self.pointwise_constraint = \
                    """pointwise_constraint='{pointwise_constraint}'""".format(
                        pointwise_constraint=self._pointwise_constraint)
                functions_required.append(self.pointwise_constraint)

            if self._bias_constraint is not None:
                self.bias_constraint = """bias_constraint='{b_const}'""".format(
                    b_const=self._bias_constraint)
                functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = SeparableConv2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class UpSampling1D(Operation):
    SIZE_PARAM = 'size_up_sampling'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.SIZE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.SIZE_PARAM)
            )

        self.size = parameters.get(self.SIZE_PARAM)

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'UpSampling1D',
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
        if self.size < 0:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.SIZE_PARAM))

        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

    def generate_code(self):
        return dedent(
            """
            {var_name} = UpSampling1D(
                name='{name}',
                size={size}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 size=self.size,
                 parent=self.parent)


class UpSampling2D(Operation):
    SIZE_PARAM = 'size_up_sampling'
    DATA_FORMAT_PARAM = 'data_format'
    INTERPOLATION_PARAM = 'interpolation'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.SIZE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.SIZE_PARAM)
            )

        self._size = parameters.get(self.SIZE_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM)
        self._interpolation = parameters.get(self.INTERPOLATION_PARAM)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.size = None
        self.data_format = None
        self.interpolation = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'UpSampling2D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False

        functions_required = []
        self.size = get_int_or_tuple(self._size)
        if self.size is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.KERNEL_SIZE_PARAM))
        functions_required.append("""size={size}""".format(size=self.size))

        if self._advanced_options:
            if self._data_format:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._interpolation:
                self.interpolation = """interpolation='{interp}'""".format(
                    interp=self._interpolation)
                functions_required.append(self.interpolation)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = UpSampling2D(
                name='{name}'{add_functions_not_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 parent=self.parent)


class UpSampling3D(Operation):
    SIZE_PARAM = 'size_up_sampling'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.SIZE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.SIZE_PARAM)
            )

        self._size = parameters.get(self.SIZE_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.size = None
        self.data_format = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'UpSampling3D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False

        functions_required = []
        self.size = get_int_or_tuple(self._size)
        if self.size is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.KERNEL_SIZE_PARAM))
        functions_required.append("""size={size}""".format(size=self.size))

        if self.advanced_options:
            if self._data_format:
                self.data_format = """data_format='{data_format}'""" \
                    .format(data_format=self._data_format)
                functions_required.append(self.data_format)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = UpSampling3D(
                name='{name}',
                size={size}{add_functions_not_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 size=self.size,
                 parent=self.parent)


class ZeroPadding1D(Operation):
    PADDING_PARAM = 'padding'
    TRAINABLE_OPTIONS_PARAM = 'trainable'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.PADDING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.PADDING_PARAM)
            )

        self._padding = parameters.get(self.PADDING_PARAM)
        self._trainable = parameters.get(self.TRAINABLE_OPTIONS_PARAM, 0)

        self.padding = None
        self.trainable = None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ''

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ZeroPadding1D',
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
        if self._padding is None:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.PADDING_PARAM))

        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.trainable = True if int(self._trainable) == 1 else \
            False

        self.padding = tuple_of_tuples(self._padding)
        if self.padding is False:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.PADDING_PARAM))

        functions_required = ["""padding={padding}""".format(padding=
                                                             self.padding)]

        if not self.trainable:
            functions_required.append('''trainable=False''')

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = ZeroPadding1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class ZeroPadding2D(Operation):
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.PADDING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.PADDING_PARAM)
            )

        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.padding = None
        self.data_format = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ZeroPadding2D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        if self._padding is not None:
            self.padding = tuple_of_tuples(self._padding)
            if self.padding is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.PADDING_PARAM))
        else:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.PADDING_PARAM)
            )

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = ZeroPadding2D(
                name='{name}',
                padding={padding}{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 padding=self.padding,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class ZeroPadding3D(Operation):
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.PADDING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.PADDING_PARAM)
            )

        self._padding = parameters.get(self.PADDING_PARAM)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.padding = None
        self.data_format = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ZeroPadding3D',
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
            self.parent.remove(python_code)

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        functions_required = []
        if self._padding is not None:
            self.padding = tuple_of_tuples(self._padding)
            if self.padding is False:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.PADDING_PARAM))
        else:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.PADDING_PARAM)
            )

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._data_format is not None:
                self.data_format = """data_format='{data_format}'""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = ZeroPadding3D(
                name='{name}',
                padding={padding}{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 padding=self.padding,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)
