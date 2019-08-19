# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.util.template_util import *


class Activation(Operation):
    ACTIVATION_PARAM = 'activation'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.ACTIVATION_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(
                self.ACTIVATION_PARAM))

        self.activation = parameters.get(self.ACTIVATION_PARAM,
                                         'linear') or 'linear'
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True
        self.parent = ""
        self.var_name = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Activation',
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

    def generate_code(self):
        return dedent(
            """
            {var_name} = Activation(
                name='{name}',
                activation='{activation}'
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 activation=self.activation,
                 parent=self.parent)


class ActivityRegularization(Operation):
    L1_PARAM = 'l1'
    L2_PARAM = 'l2'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.L1_PARAM not in parameters or self.L2_PARAM not in parameters:
            raise ValueError(
                gettext('Parameters {l1} and {l2} are required').format(
                    l1=self.L1_PARAM,
                    l2=self.L2_PARAM))

        self.l1 = parameters.get(self.L1_PARAM, 0.0) or 0.0
        self.l2 = parameters.get(self.L2_PARAM, 0.0) or 0.0
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ActivityRegularization',
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

        self.l1 = abs(self.l1)
        self.l2 = abs(self.l2)

    def generate_code(self):
        return dedent(
            """
            {var_name} = ActivityRegularization(
                name='{name}',
                l1={l1},
                l2={l2}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 l1=self.l1,
                 l2=self.l2,
                 parent=self.parent)


class Dense(Operation):
    UNITS_PARAM = 'units'
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

        if self.UNITS_PARAM not in parameters:
            raise ValueError(
                gettext('Parameter {} is required').format(self.UNITS_PARAM))

        self.units = parameters.get(self.UNITS_PARAM)
        self.activation = parameters.get(self.ACTIVATION_PARAM, 'linear')
        self.use_bias = parameters.get(self.USE_BIAS_PARAM, False) or False
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(
            self.ACTIVITY_REGULARIZER_PARAM, None)
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM,
                                              None)

        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()

        self.treatment()

        self.import_code = {'layer': 'Dense',
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

        try:
            self.units = abs(int(self.units))
        except:
            pass

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        if self.advanced_options:
            functions_required = []
            if self.kernel_initializer:
                self.kernel_initializer = """kernel_initializer=
                '{kernel_initializer}'""".format(
                    kernel_initializer=self.kernel_initializer)
                functions_required.append(self.kernel_initializer)
            if self.bias_initializer:
                self.bias_initializer = """bias_initializer=
                '{bias_initializer}'""".format(
                    bias_initializer=self.bias_initializer)
                functions_required.append(self.bias_initializer)
            if self.kernel_regularizer:
                self.kernel_regularizer = """kernel_regularizer=
                '{kernel_regularizer}'""".format(
                    kernel_regularizer=self.kernel_regularizer)
                functions_required.append(self.kernel_regularizer)
            if self.bias_regularizer:
                self.bias_regularizer = """bias_regularizer=
                '{bias_regularizer}'""".format(
                    bias_regularizer=self.bias_regularizer)
                functions_required.append(self.bias_regularizer)
            if self.activity_regularizer:
                self.activity_regularizer = """activity_regularizer=
                '{activity_regularizer}'""".format(
                    activity_regularizer=self.activity_regularizer)
                functions_required.append(self.activity_regularizer)
            if self.kernel_constraint:
                self.kernel_constraint = """kernel_constraint=
                '{kernel_constraint}'""".format(
                    kernel_constraint=self.kernel_constraint)
                functions_required.append(self.kernel_constraint)
            if self.bias_constraint:
                self.bias_constraint = """bias_constraint=
                '{bias_constraint}'""".format(
                    bias_constraint=self.bias_constraint)
                functions_required.append(self.bias_constraint)
            if self.activation:
                self.activation = """activation='{activation}'""".format(
                    activation=self.activation)
                functions_required.append(self.activation)

            self.use_bias = """use_bias='{use_bias}'""".format(
                use_bias=self.use_bias)
            functions_required.append(self.use_bias)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Dense(
                name='{task_name}',
                units={units}{add_functions_required}
            ){parent}
            """).format(var_name=self.var_name,
                        task_name=self.task_name,
                        units=self.units,
                        add_functions_required=self.add_functions_required,
                        parent=self.parent)


class Dropout(Operation):
    RATE_PARAM = 'rate'
    NOISE_SHAPE_PARAM = 'noise_shape'
    SEED_PARAM = 'seed'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.RATE_PARAM not in parameters:
            raise ValueError(
                gettext('Parameter {} is required').format(self.RATE_PARAM))

        self.rate = parameters[self.RATE_PARAM]
        self.noise_shape = parameters.get(self.NOISE_SHAPE_PARAM)
        self.seed = parameters.get(self.SEED_PARAM)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Dropout',
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

        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        if self.advanced_options:
            functions_required = []
            if self.noise_shape:
                self.noise_shape = get_int_or_tuple(self.noise_shape)
                self.noise_shape = """noise_shape='{noise_shape}'""" \
                    .format(noise_shape=self.noise_shape)
                functions_required.append(self.noise_shape)

            if self.seed:
                self.seed = """seed='{seed}'""" \
                    .format(seed=self.seed)
                functions_required.append(self.seed)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Dropout(
                name='{name}',
                rate={rate}{add_functions_not_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 rate=self.rate,
                 add_functions_not_required=self.add_functions_required,
                 parent=self.parent)


class Flatten(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM)
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Flatten',
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

        functions_required = []
        if self.data_format:
            self.data_format = get_tuple(self.data_format)
            if self.data_format is not None:
                self.data_format = """data_format={data_format}""" \
                    .format(data_format=self.data_format)
                functions_required.append(self.data_format)
            else:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.DATA_FORMAT_PARAM))

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = Flatten(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Input(Operation):
    SHAPE_PARAM = 'shape'
    BATCH_SHAPE_PARAM = 'batch_shape'
    DTYPE_PARAM = 'dtype'
    SPARSE_PARAM = 'sparse'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.shape = parameters.get(self.SHAPE_PARAM, None)
        self.batch_shape = parameters.get(self.BATCH_SHAPE_PARAM, None)
        self.dtype = parameters.get(self.DTYPE_PARAM, None)
        self.sparse = parameters.get(self.SPARSE_PARAM, 0)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self.tensor = None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.cant_be_a_tensor = ['python_code']  # slugs

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Input',
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

        self.sparse = True if int(self.sparse) == 1 else False
        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        functions_required = []
        if self.shape is not None:
            self.shape = get_tuple(self.shape)
            self.shape = """shape={shape}""".format(shape=self.shape)
            functions_required.append(self.shape)

        if self.advanced_options:
            self.sparse = """sparse={sparse}""".format(sparse=self.sparse)
            functions_required.append(self.sparse)

            if self.batch_shape is not None:
                self.batch_shape = get_tuple(self.batch_shape)
                self.batch_shape = """batch_shape={batch_shape}""" \
                    .format(batch_shape=self.batch_shape)
                functions_required.append(self.batch_shape)

            if self.dtype is not None:
                self.dtype = """dtype={dtype}""".format(dtype=self.dtype)
                functions_required.append(self.dtype)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Input(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Lambda(Operation):
    FUNCTION_PARAM = 'function'
    MASK_PARAM = 'mask'
    ARGUMENTS_PARAM = 'arguments'
    OUTPUT_SHAPE_PARAM = 'output_shape'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FUNCTION_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.FUNCTION_PARAM))

        self.function = parameters.get(self.FUNCTION_PARAM, None)
        self.mask = parameters.get(self.MASK_PARAM, None)
        self.arguments = parameters.get(self.ARGUMENTS_PARAM, None)
        self.output_shape = parameters.get(self.OUTPUT_SHAPE_PARAM,
                                           None)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True
        self.add_functions_required = ""
        self.parent = ""
        self.var_name = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Lambda',
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
            if len(self.parent) < 2:
                self.parent = '({})'.format(self.parent[0])
            else:
                self.parent = '([{}])'.format(', '.join(self.parent))
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.function is None:
            raise ValueError(
                gettext('Parameter {} is required.').format(
                    self.FUNCTION_PARAM))

        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        functions_required = ['''function={function}'''.format(function=
                                                               self.function)]

        if self.advanced_options:
            if self.mask is not None:
                functions_required.append('''mask={mask}'''.format(mask=
                                                                   self.mask))
            if self.arguments is not None:
                functions_required.append(
                    '''arguments={arguments}'''.format(arguments=
                                                       self.arguments))
            if self.output_shape is not None:
                self.output_shape = get_tuple(self.output_shape)
                functions_required.append('''output_shape={output_shape}'''.
                                          format(output_shape=self.output_shape)
                                          )

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Lambda(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Masking(Operation):
    MASK_VALUE_PARAM = 'mask_value'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.MASK_VALUE_PARAM not in parameters or self.MASK_VALUE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required').format(
                self.MASK_VALUE_PARAM))

        self.mask_value = parameters.get(self.MASK_VALUE_PARAM, 0.0) or 0.0
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Masking',
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

    def generate_code(self):
        return dedent(
            """
            {var_name} = Masking(
                name='{name}',
                mask_value={mask_value}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 mask_value=self.mask_value,
                 parent=self.parent)


class Permute(Operation):
    DIMS_PARAM = 'dims'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.DIMS_PARAM not in parameters or self.DIMS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.DIMS_PARAM))

        self.dims = parameters.get(self.DIMS_PARAM, None)
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

        self.parent = ""
        self.var_name = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Permute',
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

        if self.dims is not None:
            self.dims = get_tuple(self.dims)
        else:
            raise ValueError(gettext('Parameter {} is required. The format is: '
                                     '(x, y) or (-1, x, y)').format(
                self.DIMS_PARAM))

    def generate_code(self):
        return dedent(
            """
            {var_name} = Permute(
                name='{name}',
                dims={dims}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 dims=self.dims,
                 parent=self.parent)


class RepeatVector(Operation):
    N_PARAM = 'n'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.N_PARAM not in parameters or self.N_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.N_PARAM))

        self.n = parameters.get(self.N_PARAM, 1) or 1
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

        self.parent = ""
        self.var_name = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'RepeatVector',
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

    def generate_code(self):
        return dedent(
            """
            {var_name} = RepeatVector(
                name='{name}',
                n={n}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 n=self.n,
                 parent=self.parent)


class Reshape(Operation):
    TARGET_SHAPE_PARAM = 'target_shape'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.TARGET_SHAPE_PARAM not in parameters or self.TARGET_SHAPE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(
                self.TARGET_SHAPE_PARAM))

        self.target_shape = parameters.get(self.TARGET_SHAPE_PARAM,
                                           None)
        self.task_name = self.parameters.get('task').get('name')
        self.task_workflow_order = self.parameters.get('task').get('order')
        self.has_code = True

        self.parent = ""
        self.var_name = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Reshape',
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

        if self.target_shape is not None:
            self.target_shape = get_tuple(self.target_shape)
        else:
            raise ValueError(gettext('Parameter {} is required. The format is: '
                                     '(x, y) or (-1, x, y)').format(
                self.TARGET_SHAPE_PARAM))

    def generate_code(self):
        return dedent(
            """
            {var_name} = Reshape(
                name='{name}',
                target_shape={target_shape}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 target_shape=self.target_shape,
                 parent=self.parent)


class SpatialDropout1D(Operation):
    RATE_PARAM = 'rate'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.RATE_PARAM not in parameters or self.RATE_PARAM is None:
            raise ValueError(
                gettext('Parameter {} are required').format(self.RATE_PARAM))

        self.rate = parameters.get(self.RATE_PARAM, 0.0) or 0.0
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'SpatialDropout1D',
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

    def generate_code(self):
        return dedent(
            """
            {var_name} = SpatialDropout1D(
                name='{name}',
                rate={rate}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 rate=self.rate,
                 parent=self.parent)


class SpatialDropout2D(Operation):
    RATE_PARAM = 'rate'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.RATE_PARAM not in parameters or self.RATE_PARAM is None:
            raise ValueError(
                gettext('Parameter {} are required').format(self.RATE_PARAM))

        self.rate = parameters.get(self.RATE_PARAM, 0.0) or 0.0
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'SpatialDropout2D',
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

    def generate_code(self):
        return dedent(
            """
            {var_name} = SpatialDropout2D(
                name='{name}',
                rate={rate}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 rate=self.rate,
                 parent=self.parent)


class SpatialDropout3D(Operation):
    RATE_PARAM = 'rate'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.RATE_PARAM not in parameters or self.RATE_PARAM is None:
            raise ValueError(
                gettext('Parameter {} are required').format(self.RATE_PARAM))

        self.rate = parameters.get(self.RATE_PARAM, 0.0) or 0.0
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'SpatialDropout3D',
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

    def generate_code(self):
        return dedent(
            """
            {var_name} = SpatialDropout3D(
                name='{name}',
                rate={rate}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 rate=self.rate,
                 parent=self.parent)
