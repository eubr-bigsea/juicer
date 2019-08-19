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

        self.axis = parameters.get(self.AXIS_PARAM)
        self.momentum = parameters.get(self.MOMENTUM_PARAM)
        self.epsilon = parameters.get(self.EPSILON_PARAM)
        self.center = parameters.get(self.CENTER_PARAM)
        self.scale = parameters.get(self.SCALE_PARAM, None)
        self.beta_initializer = parameters.get(self.BETA_INITIALIZER_PARAM,
                                               None)
        self.gamma_initializer = parameters.get(self.GAMA_INITIALIZER_PARAM,
                                                None)
        self.moving_mean_initializer = parameters.get(
            self.MOVING_MEAN_VARIANCE_INITIALIZER_PARAM)
        self.moving_variance_initializer = parameters.get(
            self.MOVING_VARIANCE_INITIALIZER_PARAM, None)
        self.beta_regularizer = parameters.get(self.BETA_REGULARIZER_PARAM,
                                               None)
        self.gamma_regularizer = parameters.get(self.GAMMA_REGULARIZER_PARAM,
                                                None)
        self.beta_constraint = parameters.get(self.BETA_CONSTRAINT_PARAM,
                                              None)
        self.gamma_constraint = parameters.get(self.GAMMA_CONSTRAINT_PARAM,
                                               None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

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

        self.center = True if int(self.center) == 1 else False
        self.scale = True if int(self.scale) == 1 else False

        if self.advanced_options:
            functions_required = []
            if self.axis is None:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.AXIS_PARAM))
            else:
                self.axis = """axis={axis}""".format(axis=self.axis)
                functions_required.append(self.axis)

            self.momentum = float(self.momentum)
            if self.momentum is None or self.momentum <= 0 or self.momentum > 1:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.MOMENTUM_PARAM))
            else:
                self.momentum = """momentum={momentum}""".format(
                    momentum=self.momentum)
                functions_required.append(self.momentum)

            self.epsilon = float(self.epsilon)
            if self.epsilon is None or self.epsilon <= 0 or self.epsilon > 1:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.EPSILON_PARAM))
            else:
                self.epsilon = """epsilon={epsilon}""".format(
                    epsilon=self.epsilon)
                functions_required.append(self.epsilon)

            self.center = """center={center}""".format(center=self.center)
            functions_required.append(self.center)

            self.scale = """scale={scale}""".format(scale=self.scale)
            functions_required.append(self.scale)

            if self.beta_initializer is not None:
                self.beta_initializer = """beta_initializer='{beta_initializer}'
                """.format(beta_initializer=self.beta_initializer)
                functions_required.append(self.beta_initializer)

            if self.gamma_initializer is not None:
                self.gamma_initializer = """gamma_initializer='{gamma_initiali
                zer}'""".format(gamma_initializer=self.gamma_initializer)
                functions_required.append(self.gamma_initializer)

            if self.moving_mean_initializer is not None:
                self.moving_mean_initializer = """moving_mean_initializer=
                '{moving_mean_initializer}'""".format(
                    moving_mean_initializer=self.moving_mean_initializer)
                functions_required.append(self.moving_mean_initializer)

            if self.moving_variance_initializer is not None:
                self.moving_variance_initializer = """moving_variance_
                initializer='{moving_variance_initializer}'""".format(
                    moving_variance_initializer=self.moving_variance_initializer
                )
                functions_required.append(self.moving_variance_initializer)

            if self.beta_regularizer is not None:
                self.beta_regularizer = """beta_regularizer='{beta_regularizer}'
                """.format(beta_regularizer=self.beta_regularizer)
                functions_required.append(self.beta_regularizer)

            if self.gamma_regularizer is not None:
                self.gamma_regularizer = """gamma_regularizer='{gamma_
                regularizer}'""".format(gamma_regularizer=self.gamma_regularizer
                                        )
                functions_required.append(self.gamma_regularizer)

            if self.beta_constraint is not None:
                self.beta_constraint = """beta_constraint='{beta_constraint}'
                """.format(beta_constraint=self.beta_constraint)
                functions_required.append(self.beta_constraint)

            if self.gamma_constraint is not None:
                self.gamma_constraint = """gamma_constraint='{gamma_constraint}'
                """.format(gamma_constraint=self.gamma_constraint)
                functions_required.append(self.gamma_constraint)

            if self.kwargs is not None:
                self.kwargs = ',\n    '.join(self.kwargs.replace(' ', '')
                                             .split(','))
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