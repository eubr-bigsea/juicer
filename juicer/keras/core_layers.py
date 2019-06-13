# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


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

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.UNITS_PARAM not in parameters:
            raise ValueError(
                gettext('Parameter {} is required').format(self.UNITS_PARAM))

        self.units = parameters.get(self.UNITS_PARAM)
        self.activation = parameters.get(self.ACTIVATION_PARAM,
                                         'linear') or 'linear'
        self.use_bias = parameters.get(self.USE_BIAS_PARAM, False) or False
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None) or None
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None) or None
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None) or None
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None) or None
        self.activity_regularizer = parameters.get(
            self.ACTIVITY_REGULARIZER_PARAM, None) or None
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None) or None
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM,
                                              None) or None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Dense',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        functions_required = []
        if self.kernel_initializer:
            self.kernel_initializer = """kernel_initializer='{kernel_initializer}'""".format(
                kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)
        if self.bias_initializer:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""".format(
                bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)
        if self.kernel_regularizer:
            self.kernel_regularizer = """kernel_regularizer='{kernel_regularizer}'""".format(
                kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)
        if self.bias_regularizer:
            self.bias_regularizer = """bias_regularizer='{bias_regularizer}'""".format(
                bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer = """activity_regularizer='{activity_regularizer}'""".format(
                activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)
        if self.kernel_constraint:
            self.kernel_constraint = """kernel_constraint='{kernel_constraint}'""".format(
                kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)
        if self.bias_constraint:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""".format(
                bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = Dense(
                name='{task_name}',
                units={units}, 
                activation='{activation}', 
                use_bias={use_bias},
                {add_functions_required}
            ){parent}
            """).format(var_name=self.var_name,
                        task_name=self.task_name,
                        units=(self.units),
                        activation=(self.activation),
                        use_bias=(self.use_bias),
                        add_functions_required=self.add_functions_required,
                        parent=self.parent)


class Dropout(Operation):
    RATE_PARAM = 'rate'
    NOISE_SHAPE_PARAM = 'noise_shape'
    SEED_PARAM = 'seed'

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

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Dropout',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

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

    def generate_code(self):
        return dedent(
            """
            {var_name} = Dropout(
                name='{name}',
                rate={rate},
                {add_functions_not_required}
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

        self.treatment()

        self.import_code = {'layer': 'Flatten',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.data_format:
            self.data_format = get_tuple(self.data_format)
            if not self.data_format:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.DATA_FORMAT_PARAM))

    def generate_code(self):
        return dedent(
            """
            {var_name} = Flatten(
                name='{name}',
                data_format={data_format}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 data_format=self.data_format,
                 parent=self.parent)


class Input(Operation):
    SHAPE_PARAM = 'shape'
    BATCH_SHAPE_PARAM = 'batch_shape'
    DTYPE_PARAM = 'dtype'
    SPARSE_PARAM = 'sparse'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.shape = parameters.get(self.SHAPE_PARAM, None) or None
        self.batch_shape = parameters.get(self.BATCH_SHAPE_PARAM, None) or None
        self.dtype = parameters.get(self.DTYPE_PARAM, None) or None
        self.sparse = parameters.get(self.SPARSE_PARAM, None) or None
        self.tensor = None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.cant_be_a_tensor = ['python_code']  # slugs

        self.treatment()

        self.import_code = {'layer': 'Input',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.sparse = True if int(self.sparse) == 1 else False

        functions_required = []
        self.sparse = """sparse={sparse}""".format(sparse=self.sparse)
        functions_required.append(self.sparse)

        if self.shape is not None:
            self.shape = get_tuple(self.shape)
            self.shape = """shape={shape}""".format(shape=self.shape)
            functions_required.append(self.shape)

        if self.batch_shape is not None:
            self.batch_shape = get_tuple(self.batch_shape)
            self.batch_shape = """batch_shape={batch_shape}""" \
                .format(batch_shape=self.batch_shape)
            functions_required.append(self.batch_shape)

        if self.dtype is not None:
            self.dtype = """dtype={dtype}""".format(dtype=self.dtype)
            functions_required.append(self.dtype)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = Input(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


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

        self.treatment()

        self.import_code = {'layer': 'Activation',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))

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
                                           None) or None
        self.task_name = self.parameters.get('task').get('name')
        self.task_workflow_order = self.parameters.get('task').get('order')
        self.has_code = True

        self.parent = ""
        self.var_name = ""

        self.treatment()

        self.import_code = {'layer': 'Reshape',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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


class Permute(Operation):
    DIMS_PARAM = 'dims'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.DIMS_PARAM not in parameters or self.DIMS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.DIMS_PARAM))

        self.dims = parameters.get(self.DIMS_PARAM, None) or None
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

        self.parent = ""
        self.var_name = ""

        self.treatment()

        self.import_code = {'layer': 'Permute',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        self.treatment()

        self.import_code = {'layer': 'RepeatVector',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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


class Lambda(Operation):
    FUNCTION_PARAM = 'function'
    MASK_PARAM = 'mask'
    ARGUMENTS_PARAM = 'arguments'
    OUTPUT_SHAPE_PARAM = 'output_shape'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FUNCTION_PARAM not in parameters or self.FUNCTION_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.FUNCTION_PARAM))

        self.function = parameters.get(self.FUNCTION_PARAM, None) or None
        self.mask = parameters.get(self.MASK_PARAM, None) or None
        self.arguments = parameters.get(self.ARGUMENTS_PARAM, None) or None
        self.output_shape = parameters.get(self.OUTPUT_SHAPE_PARAM,
                                           None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True
        self.add_functions_required = ""
        self.parent = ""
        self.var_name = ""

        self.treatment()

        self.import_code = {'layer': 'Lambda',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            if len(self.parent) < 2:
                self.parent = '({})'.format(self.parent[0])
            else:
                self.parent = '([{}])'.format(', '.join(self.parent))
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if len(self.parent) < 2:
            self.parent = ''.join(self.parent)
        else:
            self.parent = ','.join(self.parent)

        if self.function is None:
            raise ValueError(
                gettext('Parameter {} is required.').format(
                    self.FUNCTION_PARAM))

        functions_required = []
        functions_required.append('''function={function}'''
                                  .format(function=self.function))
        if self.mask is not None:
            functions_required.append('''mask={mask}'''.format(mask=self.mask))
        if self.arguments is not None:
            functions_required.append(
                '''arguments={arguments}'''.format(arguments=self.arguments))
        if self.output_shape is not None:
            self.output_shape = get_tuple(self.output_shape)
            functions_required.append('''output_shape={output_shape}'''
                                      .format(output_shape=self.output_shape))

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = Lambda(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
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
                gettext('Parameter {l1} and {l2} are required').format(
                    l1=self.L1_PARAM,
                    l2=self.L2_PARAM))

        self.l1 = parameters.get(self.L1_PARAM, 0.0) or 0.0
        self.l2 = parameters.get(self.L2_PARAM, 0.0) or 0.0
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'ActivityRegularization',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        self.treatment()

        self.import_code = {'layer': 'Masking',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        self.treatment()

        self.import_code = {'layer': 'SpatialDropout1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        self.treatment()

        self.import_code = {'layer': 'SpatialDropout2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        self.treatment()

        self.import_code = {'layer': 'SpatialDropout3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
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

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'LSTM ',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.units <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive integer').format(
                    self.UNITS_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.unit_forget_bias = True if int(
            self.unit_forget_bias) == 1 else False
        self.return_sequences = True if int(
            self.return_sequences) == 1 else False
        self.return_state = True if int(self.return_state) == 1 else False
        self.go_backwards = True if int(self.go_backwards) == 1 else False
        self.stateful = True if int(self.stateful) == 1 else False
        self.unroll = True if int(self.unroll) == 1 else False

        self.implementation = int(self.implementation)

        if self.dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.DROPOUT_PARAM))
        if self.recurrent_dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.RECURRENT_DROPOUT_PARAM))

        functions_required = []
        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)
        if self.recurrent_activation is not None:
            self.recurrent_activation = """recurrent_activation='{recurrent_activation}'""" \
                .format(recurrent_activation=self.recurrent_activation)
            functions_required.append(self.recurrent_activation)
        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer='{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)
        if self.recurrent_initializer is not None:
            self.recurrent_initializer = """recurrent_initializer='{recurrent_initializer}'""" \
                .format(recurrent_initializer=self.recurrent_initializer)
            functions_required.append(self.recurrent_initializer)
        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)
        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer='{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)
        if self.recurrent_regularizer is not None:
            self.recurrent_regularizer = """recurrent_regularizer='{recurrent_regularizer}'""" \
                .format(recurrent_regularizer=self.recurrent_regularizer)
            functions_required.append(self.recurrent_regularizer)
        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer='{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)
        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer='{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)
        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint='{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)
        if self.recurrent_constraint is not None:
            self.recurrent_constraint = """recurrent_constraint='{recurrent_constraint}'""" \
                .format(recurrent_constraint=self.recurrent_constraint)
            functions_required.append(self.recurrent_constraint)
        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n               '.join(
            functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = LSTM(
                name='{name}',
                units={units},
                use_bias={use_bias},
                unit_forget_bias={unit_forget_bias},
                return_sequences={return_sequences},
                return_state={return_state},
                go_backwards={go_backwards},
                stateful={stateful},
                unroll={unroll},
                implementation={implementation},
                dropout={dropout},
                recurrent_dropout={recurrent_dropout},
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 units=self.units,
                 add_functions_required=self.add_functions_required,
                 use_bias=self.use_bias,
                 unit_forget_bias=self.unit_forget_bias,
                 return_sequences=self.return_sequences,
                 return_state=self.return_state,
                 go_backwards=self.go_backwards,
                 stateful=self.stateful,
                 unroll=self.unroll,
                 implementation=self.implementation,
                 dropout=self.dropout,
                 recurrent_dropout=self.recurrent_dropout,
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

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'SimpleRNN ',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.units <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive integer').format(
                    self.UNITS_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.return_sequences = True if int(
            self.return_sequences) == 1 else False
        self.return_state = True if int(self.return_state) == 1 else False
        self.go_backwards = True if int(self.go_backwards) == 1 else False
        self.stateful = True if int(self.stateful) == 1 else False
        self.unroll = True if int(self.unroll) == 1 else False

        if self.dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.DROPOUT_PARAM))
        if self.recurrent_dropout <= 0:
            raise ValueError(
                gettext('Parameter {} needs to be positive float').format(
                    self.RECURRENT_DROPOUT_PARAM))

        functions_required = []
        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)
        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer='{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)
        if self.recurrent_initializer is not None:
            self.recurrent_initializer = """recurrent_initializer='{recurrent_initializer}'""" \
                .format(recurrent_initializer=self.recurrent_initializer)
            functions_required.append(self.recurrent_initializer)
        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)
        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer='{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)
        if self.recurrent_regularizer is not None:
            self.recurrent_regularizer = """recurrent_regularizer='{recurrent_regularizer}'""" \
                .format(recurrent_regularizer=self.recurrent_regularizer)
            functions_required.append(self.recurrent_regularizer)
        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer='{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)
        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer='{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)
        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint='{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)
        if self.recurrent_constraint is not None:
            self.recurrent_constraint = """recurrent_constraint='{recurrent_constraint}'""" \
                .format(recurrent_constraint=self.recurrent_constraint)
            functions_required.append(self.recurrent_constraint)
        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n               '.join(
            functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = SimpleRNN(
                name='{name}',
                units={units},
                use_bias={use_bias},
                unit_forget_bias={unit_forget_bias},
                return_sequences={return_sequences},
                return_state={return_state},
                go_backwards={go_backwards},
                stateful={stateful},
                unroll={unroll},
                implementation={implementation},
                dropout={dropout},
                recurrent_dropout={recurrent_dropout},
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 units=self.units,
                 add_functions_required=self.add_functions_required,
                 use_bias=self.use_bias,
                 return_sequences=self.return_sequences,
                 return_state=self.return_state,
                 go_backwards=self.go_backwards,
                 stateful=self.stateful,
                 unroll=self.unroll,
                 dropout=self.dropout,
                 recurrent_dropout=self.recurrent_dropout,
                 parent=self.parent)


class BatchNormalization(Operation):
    AXIS_PARAM = 'axis'
    MOMENTUM_PARAM = 'momentum'
    EPSILON_PARAM = 'epsilon'
    CENTER_PARAM = 'center'
    SCALE_PARAM = 'scale'
    BETA_INITIALIZER_PARAM = 'beta_initializer'
    GAMA_INITIALIZER_PARAM = 'gamma_initializer'
    MOVING_MEAN_VARIANCE_INITIALIZER_PARAM = 'moving_mean_initializer'
    MOVING_VARIANCE_INITIALIZER_PARAM = 'moving_variance_initializer'
    BETA_REGULARIZER_PARAM = 'beta_regularizer'
    GAMMA_REGULARIZER_PARAM = 'gamma_regularizer'
    BETA_CONSTRAINT_PARAM = 'beta_constraint'
    GAMMA_CONSTRAINT_PARAM = 'gamma_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.axis = parameters.get(self.AXIS_PARAM)
        self.momentum = parameters.get(self.MOMENTUM_PARAM)
        self.epsilon = parameters.get(self.EPSILON_PARAM)
        self.center = parameters.get(self.CENTER_PARAM)
        self.scale = parameters.get(self.SCALE_PARAM, None) or None
        self.beta_initializer = parameters.get(self.BETA_INITIALIZER_PARAM,
                                               None) or None
        self.gamma_initializer = parameters.get(self.GAMA_INITIALIZER_PARAM,
                                                None) or None
        self.moving_mean_initializer = parameters.get(
            self.MOVING_MEAN_VARIANCE_INITIALIZER_PARAM)
        self.moving_variance_initializer = parameters.get(
            self.MOVING_VARIANCE_INITIALIZER_PARAM, None) or None
        self.beta_regularizer = parameters.get(self.BETA_REGULARIZER_PARAM,
                                               None) or None
        self.gamma_regularizer = parameters.get(self.GAMMA_REGULARIZER_PARAM,
                                                None) or None
        self.beta_constraint = parameters.get(self.BETA_CONSTRAINT_PARAM,
                                              None) or None
        self.gamma_constraint = parameters.get(self.GAMMA_CONSTRAINT_PARAM,
                                               None) or None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'BatchNormalization ',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.axis is None:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.AXIS_PARAM))

        self.momentum = float(self.momentum)
        if self.momentum is None or self.momentum <= 0 or self.momentum > 1:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.MOMENTUM_PARAM))

        self.epsilon = float(self.epsilon)
        if self.epsilon is None or self.epsilon <= 0 or self.epsilon > 1:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.EPSILON_PARAM))

        self.center = True if int(self.center) == 1 else False
        self.scale = True if int(self.scale) == 1 else False

        functions_required = []
        if self.beta_initializer is not None:
            self.beta_initializer = """beta_initializer='{beta_initializer}'""" \
                .format(beta_initializer=self.beta_initializer)
            functions_required.append(self.beta_initializer)

        if self.gamma_initializer is not None:
            self.gamma_initializer = """gamma_initializer=
            '{gamma_initializer}'""" \
                .format(gamma_initializer=self.gamma_initializer)
            functions_required.append(self.gamma_initializer)

        if self.moving_mean_initializer is not None:
            self.moving_mean_initializer = """moving_mean_initializer=
            '{moving_mean_initializer}'""".format(
                moving_mean_initializer=self.moving_mean_initializer)
            functions_required.append(self.moving_mean_initializer)

        if self.moving_variance_initializer is not None:
            self.moving_variance_initializer = """moving_variance_initializer=
            '{moving_variance_initializer}'""".format(
                moving_variance_initializer=self.moving_variance_initializer)
            functions_required.append(self.moving_variance_initializer)

        if self.beta_regularizer is not None:
            self.beta_regularizer = """beta_regularizer='{beta_regularizer}'""" \
                .format(beta_regularizer=self.beta_regularizer)
            functions_required.append(self.beta_regularizer)

        if self.gamma_regularizer is not None:
            self.gamma_regularizer = """gamma_regularizer=
            '{gamma_regularizer}'""".format(
                gamma_regularizer=self.gamma_regularizer)
            functions_required.append(self.gamma_regularizer)

        if self.beta_constraint is not None:
            self.beta_constraint = """beta_constraint='{beta_constraint}'""" \
                .format(beta_constraint=self.beta_constraint)
            functions_required.append(self.beta_constraint)

        if self.gamma_constraint is not None:
            self.gamma_constraint = """gamma_constraint='{gamma_constraint}'""" \
                .format(gamma_constraint=self.gamma_constraint)
            functions_required.append(self.gamma_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = BatchNormalization(
                name='{name}',
                axis={axis},
                momentum={momentum},
                epsilon={epsilon},
                center={center},
                scale={scale},
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 axis=self.axis,
                 momentum=self.momentum,
                 epsilon=self.epsilon,
                 center=self.center,
                 scale=self.scale,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class VGG16(Operation):
    INCLUDE_TOP_PARAM = 'include_top'
    WEIGHTS_PARAM = 'weights'
    INPUT_TENSOR_PARAM = 'input_tensor'
    INPUT_SHAPE_PARAM = 'input_shape'
    POOLING_PARAM = 'pooling'
    CLASSES_PARAM = 'classes'
    TRAINABLE_PARAM = 'trainable'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.include_top = parameters.get(self.INCLUDE_TOP_PARAM, None) or None
        self.weights = parameters.get(self.WEIGHTS_PARAM, None) or None
        self.input_tensor = parameters.get(self.INPUT_TENSOR_PARAM,
                                           None) or None
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None) or None
        self.pooling = parameters.get(self.POOLING_PARAM, None) or None
        self.classes = parameters.get(self.CLASSES_PARAM, None) or None
        self.trainable = parameters.get(self.TRAINABLE_PARAM, None) or None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': 'from keras.applications.vgg16 import VGG16'}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.include_top = True if int(self.include_top) == 1 else False
        self.trainable = True if int(self.trainable) == 1 else False

        if self.input_tensor:
            self.input_tensor = get_tuple(self.input_tensor)
            if not self.input_tensor:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_TENSOR_PARAM))

        if not self.include_top:
            self.input_shape = get_tuple(self.input_shape)
        else:
            self.input_shape = None
            self.pooling = None

        functions_required = []
        if self.input_tensor is not None:
            self.input_tensor = """input_tensor={input_tensor}""" \
                .format(beta_initializer=self.input_tensor)
            functions_required.append(self.input_tensor)

        if self.input_shape is not None:
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.pooling is not None:
            self.pooling = """pooling='{pooling}'""".format(
                pooling=self.pooling)
            functions_required.append(self.pooling)

        if self.classes is not None:
            self.classes = """classes={classes}""".format(classes=self.classes)
            functions_required.append(self.classes)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = VGG16(
                name='{name}',
                include_top={include_top},
                weights={weights},
                {add_functions_required}
            ){parent}
                
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 include_top=self.include_top,
                 weights=self.weights,
                 classes=self.classes,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class InceptionV3(Operation):
    INCLUDE_TOP_PARAM = 'include_top'
    WEIGHTS_PARAM = 'weights'
    INPUT_TENSOR_PARAM = 'input_tensor'
    INPUT_SHAPE_PARAM = 'input_shape'
    POOLING_PARAM = 'pooling'
    CLASSES_PARAM = 'classes'
    TRAINABLE_PARAM = 'trainable'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.include_top = parameters.get(self.INCLUDE_TOP_PARAM, None) or None
        self.weights = parameters.get(self.WEIGHTS_PARAM, None) or None
        self.input_tensor = parameters.get(self.INPUT_TENSOR_PARAM,
                                           None) or None
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None) or None
        self.pooling = parameters.get(self.POOLING_PARAM, None) or None
        self.classes = parameters.get(self.CLASSES_PARAM, None) or None
        self.trainable = parameters.get(self.TRAINABLE_PARAM, None) or None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': 'from keras.applications.inception_v3 import InceptionV3'}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.include_top = True if int(self.include_top) == 1 else False
        self.trainable = True if int(self.trainable) == 1 else False

        if self.input_tensor:
            self.input_tensor = get_tuple(self.input_tensor)
            if not self.input_tensor:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_TENSOR_PARAM))

        if not self.include_top:
            self.input_shape = get_tuple(self.input_shape)
        else:
            self.input_shape = None
            self.pooling = None

        functions_required = []
        if self.input_tensor is not None:
            self.input_tensor = """input_tensor={input_tensor}""" \
                .format(beta_initializer=self.input_tensor)
            functions_required.append(self.input_tensor)

        if self.input_shape is not None:
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.pooling is not None:
            self.pooling = """pooling='{pooling}'""".format(
                pooling=self.pooling)
            functions_required.append(self.pooling)

        if self.classes is not None:
            self.classes = """classes={classes}""".format(classes=self.classes)
            functions_required.append(self.classes)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = InceptionV3(
                name='{name}',
                include_top={include_top},
                weights={weights},
                {add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 include_top=self.include_top,
                 weights=self.weights,
                 classes=self.classes,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class PythonCode(Operation):
    CODE_PARAM = 'code'
    OUT_CODE_PARAM = 'out_code'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.code = parameters.get(self.CODE_PARAM, None) or None
        self.out_code = parameters.get(self.OUT_CODE_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')

        if self.CODE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(
                self.CODE_PARAM)
            )

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

        self.has_code = not self.out_code
        self.has_external_python_code_operation = self.out_code

    def treatment(self):
        self.out_code = True if int(self.out_code) == 1 else False

        if not self.CODE_PARAM:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.CODE_PARAM))

    def generate_code(self):
        return dedent(
            """
            
            # Begin user code - {name}
            {code}
            # End user code - {name}
            
            """
        ).format(name=self.task_name, code=self.code)


class MaxPooling1D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.pool_size = parameters.get(self.POOL_SIZE_PARAM, None) or None
        self.strides = parameters.get(self.STRIDES_PARAM, None) or None
        self.padding = parameters.get(self.PADDING_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM not in parameters or \
                self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required')
                             .format(self.POOL_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': 'MaxPooling1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        try:
            self.pool_size = int(self.pool_size)
        except:
            self.pool_size = False
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.POOL_SIZE_PARAM))

        try:
            self.strides = int(self.strides)
        except:
            self.strides = False
        if not self.strides:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        functions_required = []
        if self.strides is not None:
            self.strides = """strides={strides}""" \
                .format(strides=self.strides)
            functions_required.append(self.strides)

        if self.padding is not None:
            self.padding = """padding={padding}""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = MaxPooling1D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 pool_size=self.pool_size,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class MaxPooling2D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.pool_size = parameters.get(self.POOL_SIZE_PARAM, None) or None
        self.strides = parameters.get(self.STRIDES_PARAM, None) or None
        self.padding = parameters.get(self.PADDING_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM not in parameters or \
                self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required')
                             .format(self.POOL_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': 'MaxPooling2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        self.pool_size = get_int_or_tuple(self.pool_size)
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.POOL_SIZE_PARAM))
        else:
            self.pool_size = """pool_size={pool_size}""" \
                .format(pool_size=self.pool_size)
            functions_required.append(self.pool_size)

        if self.strides:
            self.strides = get_int_or_tuple(self.strides)
            if self.strides is None:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.STRIDES_PARAM))

        if self.strides is not None:
            self.strides = """strides={strides}""" \
                .format(strides=self.strides)
            functions_required.append(self.strides)

        if self.padding is not None:
            self.padding = """padding={padding}""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = MaxPooling2D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class MaxPooling3D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.pool_size = parameters.get(self.POOL_SIZE_PARAM, None) or None
        self.strides = parameters.get(self.STRIDES_PARAM, None) or None
        self.padding = parameters.get(self.PADDING_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM not in parameters or \
                self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required')
                             .format(self.POOL_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': 'MaxPooling3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.pool_size = get_int_or_tuple(self.pool_size)
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.POOL_SIZE_PARAM))

        self.strides = get_int_or_tuple(self.strides)
        if not self.strides:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        functions_required = []
        if self.strides is not None:
            self.strides = """strides={strides}""" \
                .format(strides=self.strides)
            functions_required.append(self.strides)

        if self.padding is not None:
            self.padding = """padding={padding}""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = MaxPooling3D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class AveragePooling1D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.pool_size = parameters.get(self.POOL_SIZE_PARAM, None) or None
        self.strides = parameters.get(self.STRIDES_PARAM, None) or None
        self.padding = parameters.get(self.PADDING_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM not in parameters or \
                self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required')
                             .format(self.POOL_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': 'AveragePooling1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        try:
            self.pool_size = int(self.pool_size)
        except:
            self.pool_size = False
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.POOL_SIZE_PARAM))

        try:
            self.strides = int(self.strides)
        except:
            self.strides = False
        if not self.strides:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        functions_required = []
        if self.strides is not None:
            self.strides = """strides={strides}""" \
                .format(strides=self.strides)
            functions_required.append(self.strides)

        if self.padding is not None:
            self.padding = """padding={padding}""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = AveragePooling1D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 pool_size=self.pool_size,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class AveragePooling2D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.pool_size = parameters.get(self.POOL_SIZE_PARAM, None) or None
        self.strides = parameters.get(self.STRIDES_PARAM, None) or None
        self.padding = parameters.get(self.PADDING_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM not in parameters or \
                self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required')
                             .format(self.POOL_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': 'AveragePooling2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.pool_size = get_int_or_tuple(self.pool_size)
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.POOL_SIZE_PARAM))

        self.strides = get_int_or_tuple(self.strides)
        if not self.strides:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        functions_required = []
        if self.strides is not None:
            self.strides = """strides={strides}""" \
                .format(strides=self.strides)
            functions_required.append(self.strides)

        if self.padding is not None:
            self.padding = """padding={padding}""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = AveragePooling2D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 pool_size=self.pool_size,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class AveragePooling3D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.pool_size = parameters.get(self.POOL_SIZE_PARAM, None) or None
        self.strides = parameters.get(self.STRIDES_PARAM, None) or None
        self.padding = parameters.get(self.PADDING_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM not in parameters or \
                self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} are required')
                             .format(self.POOL_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': 'AveragePooling3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.pool_size = get_int_or_tuple(self.pool_size)
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.POOL_SIZE_PARAM))

        self.strides = get_int_or_tuple(self.strides)
        if not self.strides:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        functions_required = []
        if self.strides is not None:
            self.strides = """strides={strides}""" \
                .format(strides=self.strides)
            functions_required.append(self.strides)

        if self.padding is not None:
            self.padding = """padding={padding}""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = AveragePooling3D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 pool_size=self.pool_size,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalMaxPooling1D(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'GlobalMaxPooling1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalMaxPooling1D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalMaxPooling2D(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'GlobalMaxPooling2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalMaxPooling2D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalMaxPooling3D(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'GlobalMaxPooling3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalMaxPooling3D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalAveragePooling1D(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'GlobalAveragePooling1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalAveragePooling1D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalAveragePooling2D(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'GlobalAveragePooling2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalAveragePooling2D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalAveragePooling3D(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'GlobalAveragePooling3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required = []
        if self.data_format is not None:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalAveragePooling3D(
                name='{name}',
                {add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Add(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Add',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = add(
                name='{name}',
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required)


class Subtract(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Subtract',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = subtract(
                name='{name}',
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required)


class Multiply(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Multiply',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs=[{inputs}]""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = multiply(
                name='{name}',
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required)


class Average(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Average',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = average(
                name='{name}',
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required)


class Maximum(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Maximum',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = maximum(
                name='{name}',
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required)


class Minimum(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Minimum',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = minimum(
                name='{name}',
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required)


class Concatenate(Operation):
    INPUTS_PARAM = 'inputs'
    AXIS_PARAM = 'axis'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.axis = parameters.get(self.AXIS_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Concatenate',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = concatenate(
                name='{name}',
                axis=axis,
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 axis=self.axis,
                 add_functions_required=self.add_functions_required)


class Dot(Operation):
    INPUTS_PARAM = 'inputs'
    AXES_PARAM = 'axes'
    NORMALIZE_PARAM = 'normalize'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None) or None
        self.axes = parameters.get(self.AXES_PARAM, None) or None
        self.normalize = parameters.get(self.NORMALIZE_PARAM, None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.treatment()

        self.import_code = {'layer': 'Dot',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.normalize = True if int(self.normalize) == 1 else False

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs is not None:
            self.inputs = """inputs={inputs}""".format(inputs=self.inputs)
            functions_required.append(self.inputs)
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        if self.axes is not None:
            self.axes = get_tuple(self.axes)
            functions_required.append(self.axes)

        if self.kwargs is not None:
            # Format kwargs
            self.kwargs = re.sub(r"^\s+|\s+$", "", self.kwargs)
            self.kwargs = re.sub(r"\s+", " ", self.kwargs)
            self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
            self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required.append(self.kwargs)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = multiply(
                name='{name},
                {normalize}=normalize,
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 normalize=self.normalize,
                 add_functions_required=self.add_functions_required)


class ModelGenerator(Operation):
    # Compile
    OPTIMIZER_PARAM = 'optimizer'
    LOSS_PARAM = 'loss'
    METRICS_PARAM = 'metrics'
    K_PARAM = 'k'
    LOSS_WEIGHTS_PARAM = 'loss_weights'
    SAMPLE_WEIGHT_MODE_PARAM = 'sample_weight_mode'
    WEIGHTED_METRICS_PARAM = 'weighted_metrics'
    TARGET_TENSORS_PARAM = 'target_tensors'
    KWARGS_PARAM = 'kwargs'

    # Fit Generator
    STEPS_PER_EPOCH_PARAM = 'steps_per_epoch'
    EPOCHS_PARAM = 'epochs'
    VERBOSE_PARAM = 'verbose'
    CALLBACKS_PARAM = 'callbacks'
    VALIDATION_DATA_PARAM = 'validation_data'
    VALIDATION_STEPS_PARAM = 'validation_steps'
    VALIDATION_FREQ_PARAM = 'validation_freq'
    CLASS_WEIGHT_PARAM = 'class_weight'
    MAX_QUEUE_SIZE_PARAM = 'max_queue_size'
    WORKERS_PARAM = 'workers'
    USE_MULTIPROCESSING_PARAM = 'use_multiprocessing'
    SHUFFLE_PARAM = 'shuffle'
    INITIAL_EPOCH_PARAM = 'initial_epoch'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        # Compile
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM, None) or None
        self.loss = parameters.get(self.LOSS_PARAM, None) or None
        self.metrics = parameters.get(self.METRICS_PARAM, None) or None
        self.k = parameters.get(self.K_PARAM, None) or None
        self.loss_weights = parameters.get(self.LOSS_WEIGHTS_PARAM,
                                           None) or None
        self.sample_weight_mode = parameters.get(self.SAMPLE_WEIGHT_MODE_PARAM,
                                                 None) or None
        self.weighted_metrics = parameters.get(self.WEIGHTED_METRICS_PARAM,
                                               None) or None
        self.target_tensors = parameters.get(self.TARGET_TENSORS_PARAM,
                                             None) or None
        self.kwargs = parameters.get(self.KWARGS_PARAM, None) or None

        # Fit Generator
        self.steps_per_epoch = parameters.get(self.STEPS_PER_EPOCH_PARAM,
                                              None) or None
        self.epochs = parameters.get(self.EPOCHS_PARAM, None) or None
        self.verbose = parameters.get(self.VERBOSE_PARAM, None) or None
        self.callbacks = parameters.get(self.CALLBACKS_PARAM, None) or None
        self.validation_data = parameters.get(self.VALIDATION_DATA_PARAM,
                                              None) or None
        self.validation_steps = parameters.get(self.VALIDATION_STEPS_PARAM,
                                               None) or None
        self.validation_freq = parameters.get(self.VALIDATION_FREQ_PARAM,
                                              None) or None
        self.class_weight = parameters.get(self.CLASS_WEIGHT_PARAM,
                                           None) or None
        self.max_queue_size = parameters.get(self.MAX_QUEUE_SIZE_PARAM,
                                             None) or None
        self.workers = parameters.get(self.WORKERS_PARAM, None) or None
        self.use_multiprocessing = parameters.get(
            self.USE_MULTIPROCESSING_PARAM, None) or None
        self.shuffle = parameters.get(self.SHUFFLE_PARAM, None) or None
        self.initial_epoch = parameters.get(self.INITIAL_EPOCH_PARAM,
                                            None) or None

        self.callback_code = ''

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required_compile = ""
        self.add_functions_required_fit_generator = ""

        self.output_task_id = self.parameters.get('task').get('id')

        if self.OPTIMIZER_PARAM not in parameters or self.optimizer is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.OPTIMIZER_PARAM))

        if self.LOSS_PARAM not in parameters or self.loss is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.LOSS_PARAM))

        if self.METRICS_PARAM not in parameters or self.metrics is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.METRICS_PARAM))

        if self.STEPS_PER_EPOCH_PARAM not in parameters or self.steps_per_epoch is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.STEPS_PER_EPOCH_PARAM))

        if self.EPOCHS_PARAM not in parameters or self.epochs is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.EPOCHS_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])

        if len(self.parents_by_port) == 0:
            raise ValueError(gettext('The operation needs the inputs.'))

        self.input_layers = []
        self.output_layers = []
        self.train_generator = None
        self.validation_generator = None

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': 'Model',
                            'preprocessing_image': None,
                            'others': None}

        self.treatment()

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        validation = ''
        for parent in self.parents_by_port:
            if str(parent[0]) == 'input-layer':
                self.input_layers.append(convert_variable_name(parent[1]))
            elif str(parent[0]) == 'output-layer':
                self.output_layers.append(convert_variable_name(parent[1]))
            elif str(parent[0]) == 'train-generator':
                self.train_generator = 'train_{var_name}' \
                    .format(var_name=convert_variable_name(parent[1]))
                if validation == '':
                    validation = convert_variable_name(parent[1])
            elif str(parent[0]) == 'validation-generator':
                validation = convert_variable_name(parent[1])

        self.validation_generator = 'validation_{var_name}' \
            .format(var_name=validation)

        if self.train_generator is None:
            raise ValueError(gettext('It is necessary to inform the training '
                                     'data.'))

        if len(self.input_layers) == 0:
            raise ValueError(gettext('It is necessary to inform the input(s) '
                                     'layer(s).'))
        if len(self.output_layers) == 0:
            raise ValueError(gettext('It is necessary to inform the output(s) '
                                     'layer(s).'))

        input_layers_vector = '['
        for input_layer in self.input_layers:
            input_layers_vector += input_layer + ', '
        input_layers_vector += ']'
        self.input_layers = input_layers_vector.replace(', ]', ']')

        output_layers_vector = '['
        for output_layer in self.output_layers:
            output_layers_vector += output_layer + ', '
        output_layers_vector += ']'
        self.output_layers = output_layers_vector.replace(', ]', ']')

        # Compile
        functions_required_compile = []
        if self.optimizer is not None:
            self.optimizer = """optimizer='{optimizer}'""" \
                .format(optimizer=self.optimizer)
            functions_required_compile.append(self.optimizer)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.optimizer))

        losses = []
        if self.loss is not None:
            for loss in self.loss:
                losses.append(loss['key'])

            self.loss = """loss={loss}""".format(loss=losses)
            functions_required_compile.append(self.loss)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.LOSS_PARAM))

        metrics = []
        if self.metrics is not None:
            for metric in self.metrics:
                metrics.append(str(metric['key']))

            self.metrics = """metrics={metrics}""" \
                .format(metrics=metrics)
            functions_required_compile.append(self.metrics)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.METRICS_PARAM))

        if 'sparse_top_k_categorical_accuracy' in metrics:
            self.k = """k={k}""" \
                .format(k=self.k)
            functions_required_compile.append(self.k)

        if self.loss_weights is not None:
            self.loss_weights = string_to_list(self.loss_weights)
            if self.loss_weights is None:
                self.loss_weights = string_to_dictionary(self.loss_weights)
                if self.loss_weights is None:
                    raise ValueError(gettext('Parameter {} is invalid.')
                                     .format(self.LOSS_WEIGHTS_PARAM))

            if self.loss_weights is not None:
                self.loss_weights = """loss_weights={loss_weights}""" \
                    .format(loss_weights=self.loss_weights)
                functions_required_compile.append(self.loss_weights)

        if self.sample_weight_mode is not None:
            if not self.sample_weight_mode == 'temporal':
                self.sample_weight_mode = string_to_list(
                    self.sample_weight_mode)
                if self.sample_weight_mode is None:
                    self.sample_weight_mode = string_to_dictionary(
                        self.sample_weight_mode)
                    if self.sample_weight_mode is None:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.SAMPLE_WEIGHT_MODE_PARAM))

            self.sample_weight_mode = """sample_weight_mode=
            {sample_weight_mode}""" \
                .format(sample_weight_mode=self.sample_weight_mode)
            functions_required_compile.append(self.sample_weight_mode)

        if self.weighted_metrics is not None:
            self.weighted_metrics = string_to_list(self.weighted_metrics)
            if self.weighted_metrics is None:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.WEIGHTED_METRICS_PARAM))
            self.weighted_metrics = """weighted_metrics={weighted_metrics}""" \
                .format(weighted_metrics=self.weighted_metrics)
            functions_required_compile.append(self.weighted_metrics)

        if self.target_tensors is not None:
            self.target_tensors = """target_tensors={target_tensors}""" \
                .format(target_tensors=self.target_tensors)
            functions_required_compile.append(self.target_tensors)

        if self.kwargs is not None:
            self.kwargs = kwargs(self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required_compile.append(self.kwargs)
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.KWARGS_PARAM))

        self.add_functions_required_compile = ',\n    ' \
            .join(functions_required_compile)

        # Fit Generator
        functions_required_fit_generator = []
        if self.train_generator is not None:
            self.train_generator = """generator={train_generator}""" \
                .format(train_generator=self.train_generator)
            functions_required_fit_generator.append(self.train_generator)

        if self.steps_per_epoch is not None:
            self.steps_per_epoch = """steps_per_epoch={steps_per_epoch}""" \
                .format(steps_per_epoch=self.steps_per_epoch)
            functions_required_fit_generator.append(self.steps_per_epoch)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.STEPS_PER_EPOCH_PARAM))

        if self.epochs is not None:
            self.epochs = """epochs={epochs}""".format(epochs=self.epochs)
            functions_required_fit_generator.append(self.epochs)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.EPOCHS_PARAM))

        if self.verbose is not None:
            self.verbose = int(self.verbose)
            self.verbose = """verbose={verbose}""" \
                .format(verbose=self.verbose)
            functions_required_fit_generator.append(self.verbose)

        # TO_DO ADD CALLBACKS CODE GENERATOR
        callbacks = '['
        if self.callbacks is not None:
            for callback in self.callbacks:
                callbacks += str(callback['key'].lower()) + ', '
                self.import_code['callbacks'].append(callback['key'])

                if callback['key'].lower() == 'tensorboard':
                    self.callback_code += 'tensorboard = TensorBoard(log_dir=' \
                                          '"/tmp/logs/{time}"' \
                                          '.format(time=time()))' + '\n'

            callbacks += ']'
            callbacks = callbacks.replace(', ]', ']')

            self.callbacks = """callbacks={callbacks}""" \
                .format(callbacks=callbacks)
            functions_required_fit_generator.append(self.callbacks)

        if self.validation_generator is not None:
            self.validation_generator = """validation_data={validation_generator}""" \
                .format(validation_generator=self.validation_generator)
            functions_required_fit_generator.append(self.validation_generator)

            if self.validation_steps is not None:
                self.validation_steps = int(self.validation_steps)
                self.validation_steps = """validation_steps={validation_steps}""" \
                    .format(validation_steps=self.validation_steps)
                functions_required_fit_generator.append(self.validation_steps)

            if self.validation_freq is not None:
                self.validation_freq = get_int_or_tuple(self.validation_freq)
                if self.validation_freq is None:
                    self.validation_freq = string_to_list(self.validation_freq)

                if self.validation_freq is not None:
                    self.validation_freq = """validation_freq={validation_freq}""" \
                        .format(validation_freq=self.validation_freq)
                    functions_required_fit_generator.append(
                        self.validation_freq)
                else:
                    raise ValueError(gettext('Parameter {} is invalid.')
                                     .format(self.VALIDATION_FREQ_PARAM))

        if self.class_weight is not None:
            self.class_weight = string_to_dictionary(self.class_weight)
            if self.class_weight is not None:
                self.class_weight = """class_weight={class_weight}""" \
                    .format(class_weight=self.class_weight)
                functions_required_fit_generator.append(self.class_weight)
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.CLASS_WEIGHT_PARAM))

        if self.max_queue_size is not None:
            self.max_queue_size = int(self.max_queue_size)
            self.max_queue_size = """max_queue_size={max_queue_size}""" \
                .format(max_queue_size=self.max_queue_size)
            functions_required_fit_generator.append(self.max_queue_size)

        if self.workers is not None:
            self.workers = int(self.workers)
            self.workers = """workers={workers}""" \
                .format(workers=self.workers)
            functions_required_fit_generator.append(self.workers)

        self.use_multiprocessing = True if int(
            self.use_multiprocessing) == 1 else False
        if self.use_multiprocessing:
            self.use_multiprocessing = """use_multiprocessing={use_multiprocessing}""" \
                .format(use_multiprocessing=self.use_multiprocessing)
            functions_required_fit_generator.append(self.use_multiprocessing)

        self.shuffle = True if int(self.shuffle) == 1 else False
        if self.shuffle:
            self.shuffle = """shuffle={shuffle}""".format(shuffle=self.shuffle)
            functions_required_fit_generator.append(self.shuffle)

        if self.initial_epoch is not None:
            self.initial_epoch = int(self.initial_epoch)
            self.initial_epoch = """initial_epoch={initial_epoch}""" \
                .format(initial_epoch=self.initial_epoch)
            functions_required_fit_generator.append(self.initial_epoch)

        self.add_functions_required_fit_generator = ',\n    ' \
            .join(functions_required_fit_generator)

    def generate_code(self):
        if not (self.train_generator and self.validation_generator):
            return dedent(
                """
                {var_name} = Model(
                    inputs={inputs},
                    outputs={outputs}
                )
                """
            ).format(var_name=self.var_name,
                     inputs=self.input_layers,
                     outputs=self.output_layers)
        else:
            return dedent(
                """
                {callback_code}
                
                {var_name} = Model(
                    inputs={inputs},
                    outputs={outputs}
                )
                {var_name}.compile(
                    {add_functions_required_compile}
                )
                {var_name}_history = {var_name}.fit_generator(
                    {add_functions_required_fit_generator}
                )
                output_task_id = '{output_task_id}'
                """
            ).format(var_name=self.var_name,
                     inputs=self.input_layers,
                     outputs=self.output_layers,
                     add_functions_required_compile=self.add_functions_required_compile,
                     add_functions_required_fit_generator=self.add_functions_required_fit_generator,
                     output_task_id=self.output_task_id,
                     callback_code=self.callback_code)


class ImageGenerator(Operation):
    FEATUREWISE_CENTER_PARAM = 'featurewise_center'
    SAMPLEWISE_CENTER_PARAM = 'samplewise_center'
    FEATUREWISE_STD_NORMALIZATION_PARAM = 'featurewise_std_normalization'
    SAMPLEWISE_STD_NORMALIZATION_PARAM = 'samplewise_std_normalization'
    ZCA_EPSILON_PARAM = 'zca_epsilon'
    ZCA_WHITENING_PARAM = 'zca_whitening'
    ROTATION_RANGE_PARAM = 'rotation_range'
    WIDTH_SHIFT_RANGE_PARAM = 'width_shift_range'
    HEIGHT_SHIFT_RANGE_PARAM = 'height_shift_range'
    BRIGHTNESS_RANGE_PARAM = 'brightness_range'
    SHEAR_RANGE_PARAM = 'shear_range'
    ZOOM_RANGE_PARAM = 'zoom_range'
    CHANNEL_SHIFT_RANGE_PARAM = 'channel_shift_range'
    FILL_MODE_PARAM = 'fill_mode'
    CVAL_PARAM = 'cval'
    HORIZONTAL_FLIP_PARAM = 'horizontal_flip'
    VERTICAL_FLIP_PARAM = 'vertical_flip'
    RESCALE_PARAM = 'rescale'
    PREPROCESSING_FUNCTION_PARAM = 'preprocessing_function'
    DATA_FORMAT_PARAM = 'data_format'
    VALIDATION_SPLIT_PARAM = 'validation_split'
    DTYPE_PARAM = 'dtype'

    TARGET_SIZE_PARAM = 'target_size'
    COLOR_MODE_PARAM = 'color_mode'
    CLASS_MODE_PARAM = 'class_mode'
    BATCH_SIZE_PARAM = 'batch_size'
    SHUFFLE_PARAM = 'shuffle'
    SEED_PARAM = 'seed'
    SUBSET_PARAM = 'subset'
    INTERPOLATION_PARAM = 'interpolation'

    TRAIN_IMAGES_PARAM = 'train_images'
    VALIDATION_IMAGES_PARAM = 'validation_images'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.featurewise_center = parameters.get(self.FEATUREWISE_CENTER_PARAM,
                                                 None) or None
        self.samplewise_center = parameters.get(self.SAMPLEWISE_CENTER_PARAM,
                                                None) or None
        self.featurewise_std_normalization = parameters.get(
            self.FEATUREWISE_STD_NORMALIZATION_PARAM, None) or None
        self.samplewise_std_normalization = parameters.get(
            self.SAMPLEWISE_STD_NORMALIZATION_PARAM, None) or None
        self.zca_epsilon = parameters.get(self.ZCA_EPSILON_PARAM, None) or None
        self.zca_whitening = parameters.get(self.ZCA_WHITENING_PARAM,
                                            None) or None
        self.rotation_range = parameters.get(self.ROTATION_RANGE_PARAM,
                                             None) or None
        self.width_shift_range = parameters.get(self.WIDTH_SHIFT_RANGE_PARAM,
                                                None) or None
        self.height_shift_range = parameters.get(self.HEIGHT_SHIFT_RANGE_PARAM,
                                                 None) or None
        self.brightness_range = parameters.get(self.BRIGHTNESS_RANGE_PARAM,
                                               None) or None
        self.shear_range = parameters.get(self.SHEAR_RANGE_PARAM, None) or None
        self.zoom_range = parameters.get(self.ZOOM_RANGE_PARAM, None) or None
        self.channel_shift_range = parameters.get(
            self.CHANNEL_SHIFT_RANGE_PARAM, None) or None
        self.fill_mode = parameters.get(self.FILL_MODE_PARAM, None) or None
        self.cval = parameters.get(self.CVAL_PARAM, None) or None
        self.horizontal_flip = parameters.get(self.HORIZONTAL_FLIP_PARAM,
                                              None) or None
        self.vertical_flip = parameters.get(self.VERTICAL_FLIP_PARAM,
                                            None) or None
        self.rescale = parameters.get(self.RESCALE_PARAM, None) or None
        self.preprocessing_function = parameters.get(
            self.PREPROCESSING_FUNCTION_PARAM, None) or None
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None
        self.validation_split = parameters.get(self.VALIDATION_SPLIT_PARAM,
                                               None) or None
        self.dtype = parameters.get(self.DTYPE_PARAM, None) or None

        self.target_size = parameters.get(self.TARGET_SIZE_PARAM, None) or None
        self.color_mode = parameters.get(self.COLOR_MODE_PARAM, None) or None
        self.class_mode = parameters.get(self.CLASS_MODE_PARAM, None) or None
        self.batch_size = parameters.get(self.BATCH_SIZE_PARAM, None) or None
        self.shuffle = parameters.get(self.SHUFFLE_PARAM, None) or None
        self.seed = parameters.get(self.SEED_PARAM, None) or None
        self.subset = parameters.get(self.SUBSET_PARAM, None) or None
        self.interpolation = parameters.get(self.INTERPOLATION_PARAM,
                                            None) or None

        self.train_images = 0
        self.validation_images = 0

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ''

        self.image_transformations = {}
        self.type = None

        if self.TARGET_SIZE_PARAM not in parameters or \
                self.TARGET_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(
                self.TARGET_SIZE_PARAM))

        if self.BATCH_SIZE_PARAM not in parameters or \
                self.BATCH_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(
                self.BATCH_SIZE_PARAM))

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None
                            }

    def treatment(self):
        parents_by_port = self.parameters.get('parents_by_port', [])
        if len(parents_by_port) == 1:
            if str(parents_by_port[0][0]) == 'train-image':
                self.train_images = parents_by_port[0]
                self.validation_images = None
            elif str(parents_by_port[0][0]) == 'validation-image':
                self.train_images = None
                self.validation_images = parents_by_port[0]

        if not (self.train_images or self.validation_images):
            raise ValueError(gettext('You need to correctly specify the '
                                     'ports for training and/or validation.'))

        if self.train_images:
            self.train_images = convert_variable_name(
                self.train_images[1]) + '_' + convert_variable_name(
                self.train_images[0])

        if self.validation_images:
            self.validation_images = convert_variable_name(
                self.validation_images[1]) + '_' + convert_variable_name(
                self.validation_images[0])

        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.target_size:
            self.target_size = get_int_or_tuple(self.target_size)
            self.image_transformations['target_size'] = self.target_size
        else:
            self.image_transformations['target_size'] = None

        if self.color_mode:
            self.image_transformations['color_mode'] = self.color_mode
        else:
            self.image_transformations['color_mode'] = None

        if self.batch_size <= 0:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.BATCH_SIZE_PARAM))

        if self.interpolation:
            self.image_transformations['interpolation'] = self.interpolation
        else:
            self.image_transformations['interpolation'] = None

        functions_required = []
        self.featurewise_center = True if int(
            self.featurewise_center) == 1 else False
        self.samplewise_center = True if int(
            self.samplewise_center) == 1 else False
        self.featurewise_std_normalization = True if int(
            self.featurewise_std_normalization) == 1 else False
        self.samplewise_std_normalization = True if int(
            self.samplewise_std_normalization) == 1 else False

        if self.zca_epsilon is not None:
            try:
                self.zca_epsilon = float(self.zca_epsilon)
            except:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.ZCA_EPSILON_PARAM))

            self.zca_epsilon = """zca_epsilon={zca_epsilon}""" \
                .format(zca_epsilon=self.zca_epsilon)
            functions_required.append(self.zca_epsilon)

        if self.featurewise_center:
            self.featurewise_center = """featurewise_center={featurewise_center}""" \
                .format(featurewise_center=self.featurewise_center)
            functions_required.append(self.featurewise_center)

        if self.samplewise_center:
            self.samplewise_center = """samplewise_center={samplewise_center}""" \
                .format(samplewise_center=self.samplewise_center)
            functions_required.append(self.samplewise_center)

        if self.featurewise_std_normalization:
            self.featurewise_std_normalization = """featurewise_std_normalization={featurewise_std_normalization}""" \
                .format(
                featurewise_std_normalization=self.featurewise_std_normalization)
            functions_required.append(self.featurewise_std_normalization)

        if self.samplewise_std_normalization:
            self.samplewise_std_normalization = """samplewise_std_normalization={samplewise_std_normalization}""" \
                .format(
                samplewise_std_normalization=self.samplewise_std_normalization)
            functions_required.append(self.samplewise_std_normalization)

        self.zca_whitening = True if int(self.zca_whitening) == 1 else False
        if self.zca_whitening:
            self.zca_whitening = """zca_whitening={zca_whitening}""" \
                .format(zca_whitening=self.zca_whitening)
            functions_required.append(self.zca_whitening)

        try:
            self.rotation_range = int(self.rotation_range)
            self.rotation_range = """rotation_range={rotation_range}""" \
                .format(rotation_range=self.rotation_range)
            functions_required.append(self.rotation_range)
        except:
            raise ValueError(gettext(',\nParameter {} is invalid.')
                             .format(self.ROTATION_RANGE_PARAM))

        self.width_shift_range = string_to_int_float_list(
            self.width_shift_range)
        if self.width_shift_range is not None:
            self.width_shift_range = """width_shift_range={width_shift_range}""" \
                .format(width_shift_range=self.width_shift_range)
            functions_required.append(self.width_shift_range)
        else:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.WIDTH_SHIFT_RANGE_PARAM))

        self.height_shift_range = string_to_int_float_list(
            self.height_shift_range)
        if self.height_shift_range is not None:
            self.height_shift_range = """height_shift_range={height_shift_range}""" \
                .format(height_shift_range=self.height_shift_range)
            functions_required.append(self.height_shift_range)
        else:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.HEIGHT_SHIFT_RANGE_PARAM))

        if self.brightness_range is not None:
            self.brightness_range = string_to_list(self.brightness_range)
            if self.brightness_range is not None and \
                    len(self.brightness_range) == 2:
                self.brightness_range = """brightness_range={brightness_range}""" \
                    .format(brightness_range=self.brightness_range)
                functions_required.append(self.brightness_range)
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.BRIGHTNESS_RANGE_PARAM))

        try:
            self.shear_range = float(self.shear_range)
        except:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.SHEAR_RANGE_PARAM))
        self.shear_range = """shear_range={shear_range}""" \
            .format(shear_range=self.shear_range)
        functions_required.append(self.shear_range)

        self.zoom_range = string_to_list(self.zoom_range)
        if self.zoom_range and len(self.zoom_range) <= 2:
            if len(self.zoom_range) == 1:
                self.zoom_range = float(self.zoom_range[0])

            self.zoom_range = """zoom_range={zoom_range}""" \
                .format(zoom_range=self.zoom_range)
            functions_required.append(self.zoom_range)
        else:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.ZOOM_RANGE_PARAM))

        try:
            self.channel_shift_range = float(self.channel_shift_range)
            self.channel_shift_range = """channel_shift_range={channel_shift_range}""" \
                .format(channel_shift_range=self.channel_shift_range)
            functions_required.append(self.channel_shift_range)
        except:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.CHANNEL_SHIFT_RANGE_PARAM))

        if self.fill_mode:
            self.fill_mode = """fill_mode='{fill_mode}'""" \
                .format(fill_mode=self.fill_mode)
            functions_required.append(self.fill_mode)

        if self.fill_mode == 'constant':
            try:
                self.cval = float(self.cval)
            except:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.CVAL_PARAM))
            self.cval = """cval={cval}""".format(cval=self.cval)
            functions_required.append(self.cval)

        self.horizontal_flip = True if int(self.horizontal_flip) == 1 else False
        if self.horizontal_flip:
            self.horizontal_flip = """horizontal_flip={horizontal_flip}""" \
                .format(horizontal_flip=self.horizontal_flip)
            functions_required.append(self.horizontal_flip)

        self.vertical_flip = True if int(self.vertical_flip) == 1 else False
        if self.vertical_flip:
            self.vertical_flip = """vertical_flip={vertical_flip}""" \
                .format(vertical_flip=self.vertical_flip)
            functions_required.append(self.vertical_flip)

        if self.rescale is not None:
            self.rescale = rescale(self.rescale)
            if self.rescale is not None:
                self.rescale = """rescale={rescale}""" \
                    .format(rescale=self.rescale)
                functions_required.append(self.rescale)

        '''TO_DO - ADD preprocessing_function IN THE FUTURE'''

        if self.data_format:
            self.data_format = """data_format={data_format}""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        # In case of the operation is creating the image data
        if self.train_images:
            self.validation_split = abs(float(self.validation_split))
            if self.validation_split > 0:
                self.validation_split = """validation_split={validation_split}""" \
                    .format(validation_split=self.validation_split)
                functions_required.append(self.validation_split)

        if self.dtype is not None:
            self.dtype = """dtype={dtype}""" \
                .format(dtype=self.dtype)
            functions_required.append(self.dtype)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        if self.train_images:
            if self.validation_split > 0:
                return dedent(
                    """
                    {var_name}_datagen = ImageDataGenerator(
                    {add_functions_required}
                    )
                    
                    train_{var_name} = {dataset_generator}(
                        tar_path={train_path},
                        batch_size={batch_size},
                        image_data_generator={var_name}_datagen, 
                        seed={seed},
                        split={split},
                        image_transformations={image_transformations},
                        subset='training'
                    )
                    train_{var_name} = train_{var_name}.read()
                        
                    validation_{var_name} = {dataset_generator}(
                        tar_path={validation_path},
                        batch_size={batch_size},
                        image_data_generator={var_name}_datagen, 
                        seed={seed},
                        split={split},
                        image_transformations={image_transformations},
                        subset='validation'
                    )
                    validation_{var_name} = validation_{var_name}.read()
                    """
                ).format(var_name=self.var_name,
                         add_functions_required=self.add_functions_required,
                         dataset_generator='{parent}_dataset_generator'
                         .format(parent=self.parents[0]),
                         train_path='{parent}_train_images'
                         .format(parent=self.parents[0]),
                         validation_path='{parent}_validation_images'
                         .format(parent=self.parents[0]),
                         batch_size=self.batch_size,
                         seed=self.seed,
                         split=self.validation_split,
                         image_transformations=self.image_transformations)
            else:
                return dedent(
                    """
                    {var_name}_datagen = ImageDataGenerator(
                        {add_functions_required}
                    )
                    
                    train_{var_name} = {dataset_generator}(
                        tar_path={train_path},
                        batch_size={batch_size},
                        image_data_generator={var_name}_datagen, 
                        seed={seed},
                        image_transformations={image_transformations},
                        subset='training'
                    )
                    train_{var_name} = train_{var_name}.read()
                    
                    validation_{var_name} = None
                    """
                ).format(var_name=self.var_name,
                         add_functions_required=self.add_functions_required,
                         dataset_generator='{parent}_dataset_generator'
                         .format(parent=self.parents[0]),
                         train_path='{parent}_train_images'
                         .format(parent=self.parents[0]),
                         batch_size=self.batch_size,
                         seed=self.seed,
                         image_transformations=self.image_transformations)

        if self.validation_images:
            return dedent(
                """
                {var_name}_datagen = ImageDataGenerator(
                    {add_functions_required}
                )
                
                validation_{var_name} = {dataset_generator}(
                    tar_path={validation_path},
                    batch_size={batch_size},
                    image_data_generator={var_name}_datagen, 
                    seed={seed},
                    image_transformations={image_transformations},
                    subset='validation'
                )
                validation_{var_name} = validation_{var_name}.read()
                """
            ).format(var_name=self.var_name,
                     add_functions_required=self.add_functions_required,
                     dataset_generator='{parent}_dataset_generator'
                     .format(parent=self.parents[0]),
                     validation_path='{parent}_validation_images'
                     .format(parent=self.parents[0]),
                     batch_size=self.batch_size,
                     seed=self.seed,
                     image_transformations=self.image_transformations)


class ImageReader(Operation):
    TRAIN_IMAGES_PARAM = 'train_images'
    VALIDATION_IMAGES_PARAM = 'validation_images'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.TRAIN_IMAGES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.TRAIN_IMAGES_PARAM))

        self.train_images = parameters.get(self.TRAIN_IMAGES_PARAM,
                                           None) or None
        self.validation_images = parameters.get(self.VALIDATION_IMAGES_PARAM,
                                                None) or None

        self.has_code = True
        self.var_name = ""
        self.task_name = self.parameters.get('task').get('name')

        supported_formats = ('HAR_IMAGE_FOLDER', 'TAR_IMAGE_FOLDER')

        if self.train_images != 0 and self.train_images is not None:
            self.metadata_train = self.get_data_source(
                data_source_id=self.train_images)

            if self.metadata_train.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported image format: {}').format(
                    self.metadata_train.get('format')))

            self.format = self.metadata_train.get('format')

        if self.validation_images != 0 and self.validation_images is not None:
            self.metadata_validation = self.get_data_source(
                data_source_id=self.validation_images)

            if self.metadata_validation.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported image format: {}').format(
                    self.metadata_validation.get('format')))

            self.format = self.metadata_validation.get('format')
        else:
            self.metadata_validation = None

        if self.metadata_validation is not None:
            if not (self.metadata_train.get('format') ==
                    self.metadata_validation.get('format')):
                raise ValueError(gettext('Training and validation images files '
                                         'are in different formats.'))

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': 'ImageDataGenerator',
                            'others': 'from juicer.keras.persistence import '
                                      '{}'.format(self.get_generator_name())
                            }

    def get_generator_name(self):
        if self.format == 'HAR_IMAGE_FOLDER':
            return 'har_image_generator'
        elif self.format == 'TAR_IMAGE_FOLDER':
            return 'ArchiveImageGenerator'

    def get_data_source(self, data_source_id):
        # Retrieve metadata from Limonero.
        limonero_config = \
        self.parameters['configuration']['juicer']['services']['limonero']

        metadata = limonero_service.get_data_source_info(
            limonero_config['url'], str(limonero_config['auth_token']),
            data_source_id)

        if not metadata.get('url'):
            raise ValueError(
                gettext('Incorrect data source configuration (empty url)'))

        return metadata

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.train_images is not None:
            self.train_images = """'{storage_url}{file_url}'""".format(
                storage_url=self.metadata_train.get('storage').get('url'),
                file_url=self.metadata_train.get('url')
            )

        if self.validation_images is not None:
            self.validation_images = """'{storage_url}{file_url}'""".format(
                storage_url=self.metadata_validation.get('storage').get('url'),
                file_url=self.metadata_validation.get('url')
            )

        self.train_images = self.train_images.replace('file://','')
        self.validation_images = self.validation_images.replace('file://','')

    def generate_code(self):
        if self.train_images and self.validation_images:
            return dedent(
                """                
                {var_name}_train_images = {train_images}
                {var_name}_validation_images = {validation_images}
                
                {var_name}_dataset_generator = {dataset_generator}
                """.format(var_name=self.var_name,
                           dataset_generator=self.get_generator_name(),
                           train_images=self.train_images,
                           validation_images=self.validation_images)
            )

        if self.train_images:
            return dedent(
                """
                {var_name}_train_images = {train_images}
                
                {var_name}_dataset_generator = {dataset_generator}
                """.format(var_name=self.var_name,
                           dataset_generator=self.get_generator_name(),
                           train_images=self.train_images)
            )


'''
class Hyperparameters(Operation):
    NUMBER_OF_EPOCHS_PARAM = 'number_of_epochs'
    BATCH_SIZE_PARAM = 'batch_size'
    LOSS_PARAM = 'loss'
    OPTIMIZER_PARAM = 'optimizer'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.LOSS_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.LOSS_PARAM))

        if self.OPTIMIZER_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.OPTIMIZER_PARAM))

        self.optimizer = parameters.get(self.OPTIMIZER_PARAM)
        self.number_of_epochs = parameters.get(self.NUMBER_OF_EPOCHS_PARAM)
        self.batch_size = parameters.get(self.BATCH_SIZE_PARAM)
        self.loss = parameters.get(self.LOSS_PARAM)

        self.has_code = True

        self.treatment()

    def treatment(self):
        if self.number_of_epochs <= 0:
            raise ValueError(gettext('Parameter {} requires a valid value').format(self.NUMBER_OF_EPOCHS_PARAM))

    def generate_code(self):
        return dedent(
            """
            loss_function = "{loss_function}"
            optimizer_function = "{optimizer_function}"
            """.format(loss_function=self.loss, optimizer_function=self.optimizer))
'''
