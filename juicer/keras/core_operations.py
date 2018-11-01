# -*- coding: utf-8 -*-
from gettext import gettext
from textwrap import dedent

from juicer.operation import Operation

class DenseOperation(Operation):
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
            raise ValueError(gettext('Parameter {} is required').format(self.UNITS_PARAM))

        self.units = parameters.get(self.UNITS_PARAM)
        self.activation = parameters.get(self.ACTIVATION_PARAM,
                                         'linear') or 'linear'
        self.use_bias = parameters.get(self.USE_BIAS_PARAM, False) or False
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM, None) or None
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM, None) or None
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM, None) or None
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM, None) or None
        self.activity_regularizer = parameters.get(self.ACTIVITY_REGULARIZER_PARAM, None) or None
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM, None) or None
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None) or None
        self.add_functions_not_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

        self.treatment()

    def treatment(self):
        if int(self.units) < 0:
            self.units = abs(self.units)
        if self.use_bias == '1':
            self.use_bias = True
        else:
            self.use_bias = False

        functions_not_required = []
        if self.kernel_initializer:
            self.kernel_initializer = """kernel_initializer='{kernel_initializer}'""".format(kernel_initializer=self.kernel_initializer)
            functions_not_required.append(self.kernel_initializer)
        if self.bias_initializer:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""".format(bias_initializer=self.bias_initializer)
            functions_not_required.append(self.bias_initializer)
        if self.kernel_regularizer:
            self.kernel_regularizer = """kernel_regularizer='{kernel_regularizer}'""".format(kernel_regularizer=self.kernel_regularizer)
            functions_not_required.append(self.kernel_regularizer)
        if self.bias_regularizer:
            self.bias_regularizer = """bias_regularizer='{bias_regularizer}'""".format(bias_regularizer=self.bias_regularizer)
            functions_not_required.append(self.bias_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer = """activity_regularizer='{activity_regularizer}'""".format(activity_regularizer=self.activity_regularizer)
            functions_not_required.append(self.activity_regularizer)
        if self.kernel_constraint:
            self.kernel_constraint = """kernel_constraint='{kernel_constraint}'""".format(kernel_constraint=self.kernel_constraint)
            functions_not_required.append(self.kernel_constraint)
        if self.bias_constraint:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""".format(bias_constraint=self.bias_constraint)
            functions_not_required.append(self.bias_constraint)

        # Mount
        length = len(functions_not_required)
        for i in range(0, length):
            if not i == 0:
                self.add_functions_not_required += '            '
            if i == (length - 1):
                self.add_functions_not_required += functions_not_required[i]
            else:
                self.add_functions_not_required += functions_not_required[i] + ",\n"

    def generate_code(self):
        return dedent("""
        model.add(Dense(name='{name}',
                    units={units}, 
                    activation='{activation}', 
                    use_bias={use_bias},
                    {add_functions_not_required}))
        """).format(name=self.task_name,
                    units=(self.units),
                    activation=(self.activation),
                    use_bias=(self.use_bias),
                    add_functions_not_required=self.add_functions_not_required)


class DropoutOperation(Operation):
    RATE_PARAM = 'rate'
    NOISE_SHAPE_PARAM = 'noise_shape'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.RATE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.RATE_PARAM))

        self.rate = parameters[self.RATE_PARAM]
        self.noise_shape = parameters.get(self.NOISE_SHAPE_PARAM)
        self.seed = parameters.get(self.SEED_PARAM)
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

    def generate_code(self):
        return dedent(
            """
            model.add(Dropout(name='{name}', rate={rate}, noise_shape={noise_shape}, seed={seed}))
            """
        ).format(name=self.task_name, rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)


class FlattenOperation(Operation):
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.data_format = parameters.get(self.DATA_FORMAT_PARAM)
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

    def generate_code(self):
        return dedent(
            """
            model.add(Flatten(name='{name}', data_format={data_format}))
            """
        ).format(name=self.task_name, data_format=self.data_format)


class OptimizerOperation(Operation):
    OPTIMIZER_PARAM = 'optimizer'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.OPTIMIZER_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.OPTIMIZER_PARAM))

        self.optimizer = parameters.get(self.OPTIMIZER_PARAM)
        self.has_code = True

    def generate_code(self):
        return dedent(
            """
            optimizer_function = "{optimizer_function}"
            """.format(optimizer_function=self.optimizer)
        )


class LossOperation(Operation):
    LOSS_PARAM = 'loss'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.LOSS_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.LOSS_PARAM))

        self.loss = parameters.get(self.LOSS_PARAM)
        self.has_code = True

    def generate_code(self):
        return dedent(
            """
            loss_function = "{loss_function}"
            """.format(loss_function=self.loss)
        )


class InputOperation(Operation):
    DATASET_PARAM = 'dataset'
    TRAIN_VALIDATION_TEST_SPLIT_PARAM = 'train_validation_test_split'
    USE_K_FOLD_CROSS_VALIDATION_PARAM = 'use_k_fold_cross_validation'
    PERCENT_OF_TRAIN_DATA_PARAM = 'percent_of_train_data'
    SHUFFLE_PARAM = 'shuffle_data'
    LOAD_DATASET_IN_MEMORY_PARAM = 'load_dataset_in_memory'
    SEED_PARAM = 'seed'


    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.DATASET_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.DATASET_PARAM))

        self.dataset = parameters.get(self.DATASET_PARAM, None)
        self.train_validation_test_split = parameters.get(self.TRAIN_VALIDATION_TEST_SPLIT_PARAM)
        self.use_k_fold_cross_validation = parameters.get(self.USE_K_FOLD_CROSS_VALIDATION_PARAM)
        self.percent_of_train_data = parameters.get(self.PERCENT_OF_TRAIN_DATA_PARAM)
        self.shuffle_data = parameters.get(self.SHUFFLE_PARAM)
        self.load_dataset_in_memory = parameters.get(self.LOAD_DATASET_IN_MEMORY_PARAM)
        self.seed = parameters.get(self.SEED_PARAM)

        self.is_image = False
        self.has_code = True

        self.treatment()

    def treatment(self):
        # TO_DO - REMOVE
        if self.dataset:
            if '.csv' not in self.dataset:
                self.is_image = True
            else:
                self.is_image = False
        else:
            raise ValueError(gettext('Parameter {} is required').format(self.dataset))

        if self.use_k_fold_cross_validation == '1':
            self.use_k_fold_cross_validation = True
        else:
            self.use_k_fold_cross_validation = False

        if self.shuffle_data == '1':
            self.shuffle_data = True
        else:
            self.shuffle_data = False

        if len(self.train_validation_test_split) == 0 and not self.use_k_fold_cross_validation:
            raise ValueError(gettext('Parameter {} needs to be like 60%-20%-20%').format(self.train_validation_test_split))
        elif self.use_k_fold_cross_validation:
            if self.percent_of_train_data >= 100 or self.percent_of_train_data < 0:
                raise ValueError(gettext('Parameter {} needs to be 0 < x < 100').format(self.PERCENT_OF_TRAIN_DATA))
        else:
            import re
            pattern = re.compile("^[0-9]{2}%-[0-9]{2}%-[0-9]{2}%$")
            is_pattern = False
            if bool(pattern.match(self.train_validation_test_split)):
                self.train_validation_test_split = self.train_validation_test_split.replace('%', '').split('-')
                if len(self.train_validation_test_split) == 3:
                    validate = 0
                    for item in self.train_validation_test_split:
                        validate += int(item)
                    if validate == 100:
                        is_pattern = True

            if not is_pattern:
                raise ValueError(gettext('Parameter {} needs to be like 60%-20%-20%').format(self.train_validation_test_split))

    def generate_code(self):
        return dedent("""
            np.random.seed({seed})
            dataset = np.loadtxt("{dataset}", delimiter=",")
            input_var = dataset[:,0:8]
            output_var = dataset[:,8]
            """).format(seed=self.seed, dataset=self.dataset)


class OutputOperation(Operation):

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.output_task_id = self.parameters.get('task').get('id')
        self.has_code = True

    def generate_code(self):
        return dedent(
            """
            output_task_id = '{output_task_id}'
            """).format(output_task_id=self.output_task_id)


class ActivationOperation(Operation):
    ACTIVATION_PARAM = 'activation'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.ACTIVATION_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(self.ACTIVATION_PARAM))

        self.activation = parameters.get(self.ACTIVATION_PARAM, 'linear') or 'linear'
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

    def generate_code(self):
        return dedent(
            """
            model.add(Activation(name='{name}', activation='{activation}'))
            """
        ).format(name=self.task_name, activation=self.activation)


class ReshapeOperation(Operation):
    TARGET_SHAPE_PARAM = 'target_shape'
    INPUT_SHAPE_PARAM = 'input_shape'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.TARGET_SHAPE_PARAM not in parameters or self.TARGET_SHAPE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(self.TARGET_SHAPE_PARAM))

        self.target_shape = parameters.get(self.TARGET_SHAPE_PARAM, None) or None
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None) or None
        self.task_name = self.parameters.get('task').get('name')
        self.task_workflow_order = self.parameters.get('task').get('order')
        self.has_code = True
        self.add_input_shape = ''

        self.treatment()

    def treatment(self):
        if self.target_shape is not None:
            import re
            regex = re.compile('\((\s*\-{0,1}\d+\s*,){0,1}(\s*\d+\s*,\s*\d+\s*)\)')
            if regex.match(self.target_shape) is None:
                raise ValueError(gettext('Parameter {} is required. The format is: '
                                         '(x, y) or (-1, x, y)').format(self.TARGET_SHAPE_PARAM))

            '''
            if self.input_shape is not None:
                if self.task_workflow_order in [1, 2]:
                    regex = re.compile('\(\s*\d+\s*,\s*\)')
                else:
                    regex = re.compile('\(\s*\d+\s*,\s*\d+\s*\)')
                if regex.match(self.input_shape) is None:
                    raise ValueError(gettext('Parameter {} is required. The format is: '
                                             '(x,y)').format(self.INPUT_SHAPE_PARAM))
                else:
                    self.add_input_shape = ', input_shape=' + self.input_shape
            '''

    def generate_code(self):
        return dedent(
            """
            model.add(Reshape(name='{name}', target_shape={target_shape}{add_input_shape}))
            """
        ).format(name=self.task_name, target_shape=self.target_shape, add_input_shape=self.add_input_shape)


class PermuteOperation(Operation):
    DIMS_PARAM = 'dims'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.DIMS_PARAM not in parameters or self.DIMS_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(self.DIMS_PARAM))

        self.dims = parameters.get(self.DIMS_PARAM, None) or None
        self.task_name = self.parameters.get('task').get('name')
        self.has_code = True

        self.treatment()

    def treatment(self):
        if self.dims is not None:
            import re
            regex = re.compile('\(\s*\d+\s*,\s*\d+\s*\)')
            if regex.match(self.dims) is None:
                raise ValueError(gettext('Parameter {} is required. The format is: '
                                         '(x, y)').format(self.DIMS_PARAM))

    def generate_code(self):
        return dedent(
            """
            model.add(Permute(name='{name}', dims={dims}))
            """
        ).format(name=self.task_name, dims=self.dims)
