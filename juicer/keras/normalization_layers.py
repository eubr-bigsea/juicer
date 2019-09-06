# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


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
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.axis = int(parameters.get(self.AXIS_PARAM, -1))
        self.momentum = float(parameters.get(self.MOMENTUM_PARAM, 0.99))
        self.epsilon = float(parameters.get(self.EPSILON_PARAM, 0.001))
        self._center = parameters.get(self.CENTER_PARAM)
        self._scale = parameters.get(self.SCALE_PARAM, None)
        self._beta_initializer = parameters.get(self.BETA_INITIALIZER_PARAM,
                                                None)
        self._gamma_initializer = parameters.get(self.GAMA_INITIALIZER_PARAM,
                                                 None)
        self._moving_mean_initializer = parameters.get(
            self.MOVING_MEAN_VARIANCE_INITIALIZER_PARAM)
        self._moving_variance_initializer = parameters.get(
            self.MOVING_VARIANCE_INITIALIZER_PARAM, None)
        self._beta_regularizer = parameters.get(self.BETA_REGULARIZER_PARAM,
                                                None)
        self._gamma_regularizer = parameters.get(self.GAMMA_REGULARIZER_PARAM,
                                                 None)
        self._beta_constraint = parameters.get(self.BETA_CONSTRAINT_PARAM,
                                               None)
        self._gamma_constraint = parameters.get(self.GAMMA_CONSTRAINT_PARAM,
                                                None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.center = None
        self.scale = None
        self.beta_initializer = None
        self.gamma_initializer = None
        self.moving_mean_initializer = None
        self.moving_variance_initializer = None
        self.beta_regularizer = None
        self.gamma_regularizer = None
        self.beta_constraint = None
        self.gamma_constraint = None
        self.kwargs = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'BatchNormalization ',
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

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False

        if self.advanced_options:
            functions_required = []
            if self.axis is None:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.AXIS_PARAM))
            else:
                functions_required.append("""axis={axis}""".format(
                    axis=self.axis))

            if self.momentum is None or self.momentum <= 0 or self.momentum > 1:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.MOMENTUM_PARAM))
            else:
                functions_required.append("""momentum={momentum}""".format(
                    momentum=self.momentum))

            if self.epsilon is None or self.epsilon <= 0 or self.epsilon > 1:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.EPSILON_PARAM))
            else:
                functions_required.append("""epsilon={epsilon}""".format(
                    epsilon=self.epsilon))

            self.center = True if int(self._center) == 1 else False
            functions_required.append("""center={center}""".format(
                center=self.center))

            self.scale = True if int(self._scale) == 1 else False
            functions_required.append("""scale={scale}""".format(
                scale=self.scale))

            if self._beta_initializer is not None:
                self.beta_initializer = \
                    """beta_initializer='{beta_initializer}'""".format(
                        beta_initializer=self._beta_initializer)
                functions_required.append(self.beta_initializer)

            if self._gamma_initializer is not None:
                self.gamma_initializer = """gamma_initializer='{g}'""".format(
                    g=self._gamma_initializer)
                functions_required.append(self.gamma_initializer)

            if self._moving_mean_initializer is not None:
                self.moving_mean_initializer = \
                    """moving_mean_initializer='{mmi}'""".format(
                        mmi=self._moving_mean_initializer)
                functions_required.append(self.moving_mean_initializer)

            if self._moving_variance_initializer is not None:
                self.moving_variance_initializer = \
                    """moving_variance_initializer='{mvi}'""".format(
                        mvi=self._moving_variance_initializer
                )
                functions_required.append(self.moving_variance_initializer)

            if self._beta_regularizer is not None:
                self.beta_regularizer = """beta_regularizer='{b}'""".format(
                    b=self._beta_regularizer)
                functions_required.append(self.beta_regularizer)

            if self._gamma_regularizer is not None:
                self.gamma_regularizer = """gamma_regularizer='{g}'""".format(
                    g=self._gamma_regularizer)
                functions_required.append(self.gamma_regularizer)

            if self._beta_constraint is not None:
                self.beta_constraint = """beta_constraint='{b}'""".format(
                    b=self._beta_constraint)
                functions_required.append(self.beta_constraint)

            if self._gamma_constraint is not None:
                self.gamma_constraint = """gamma_constraint='{g}'""".format(
                    g=self._gamma_constraint)
                functions_required.append(self.gamma_constraint)

            if self.kwargs is not None:
                self.kwargs = ',\n    '.join(self._kwargs.replace(
                    ' ', '').split(','))
                functions_required.append(self.kwargs)

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = BatchNormalization(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)
