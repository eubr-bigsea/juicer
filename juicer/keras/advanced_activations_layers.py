# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.util.template_util import *


class LeakyReLU(Operation):
    ALPHA_PARAM = 'alpha'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.alpha = float(parameters.get(self.ALPHA_PARAM))
        if not self.alpha >= 0.0:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.ALPHA_PARAM)
            )

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'LeakyReLU',
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
            {var_name} = LeakyReLU(
                name='{name}',
                alpha={alpha}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 alpha=self.alpha,
                 parent=self.parent)


class PReLU(Operation):
    ALPHA_INITIALIZER_PARAM = 'alpha_initializer'
    ALPHA_REGULARIZER_PARAM = 'alpha_regularizer'
    ALPHA_CONSTRAINT_PARAM = 'alpha_constraint'
    SHARED_AXES_PARAM = 'shared_axes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.alpha_initializer = parameters.get(self.ALPHA_INITIALIZER_PARAM, None)
        self.alpha_regularizer = parameters.get(self.ALPHA_REGULARIZER_PARAM, None)
        self.alpha_constraint = parameters.get(self.ALPHA_CONSTRAINT_PARAM, None)
        self.shared_axes = parameters.get(self.SHARED_AXES_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'PReLU',
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
        if self.alpha_initializer:
            functions_required.append("""alpha_initializer={alpha_initializer}""".format(
                alpha_initializer=self.alpha_initializer))

        if self.alpha_regularizer:
            functions_required.append("""alpha_regularizer={alpha_regularizer}""".format(
                alpha_regularizer=self.alpha_regularizer))

        if self.alpha_constraint:
            functions_required.append("""alpha_constraint={alpha_constraint}""".format(
                alpha_constraint=self.alpha_constraint))

        if self.shared_axes:
            functions_required.append("""shared_axes={shared_axes}""".format(
                shared_axes=string_to_list(self.shared_axes)))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = PReLU(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class ELU(Operation):
    ALPHA_PARAM = 'alpha'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.alpha = float(parameters.get(self.ALPHA_PARAM))

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ELU',
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
            {var_name} = ELU(
                name='{name}',
                alpha={alpha}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 alpha=self.alpha,
                 parent=self.parent)


class ThresholdedReLU(Operation):
    THETA_PARAM = 'theta'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.theta = float(parameters.get(self.THETA_PARAM))
        if not self.theta >= 0.0:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.THETA_PARAM)
            )

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ThresholdedReLU',
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
            {var_name} = ThresholdedReLU(
                name='{name}',
                theta={theta}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 theta=self.theta,
                 parent=self.parent)


class Softmax(Operation):
    AXIS_PARAM = 'axis'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.axis = int(parameters.get(self.AXIS_PARAM, -1))

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'Softmax',
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
            {var_name} = Softmax(
                name='{name}',
                axis={axis}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 axis=self.axis,
                 parent=self.parent)


class ReLU(Operation):
    MAX_VALUE_PARAM = 'max_value'
    NEGATIVE_SLOPE_PARAM = 'negative_slope'
    THRESHOLD_PARAM = 'threshold'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.max_value = float(parameters.get(self.MAX_VALUE_PARAM))
        if not self.max_value >= 0.0:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.MAX_VALUE_PARAM)
            )
        self.negative_slope = float(parameters.get(self.NEGATIVE_SLOPE_PARAM))
        if not self.negative_slope >= 0.0:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.NEGATIVE_SLOPE_PARAM)
            )
        self.threshold = float(parameters.get(self.THRESHOLD_PARAM))

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'ReLU',
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
        if self.max_value:
            functions_required.append("""max_value={max_value}""".format(
                max_value=self.max_value))

        if self.negative_slope:
            functions_required.append("""negative_slope={negative_slope}""".format(
                negative_slope=self.negative_slope))

        if self.threshold:
            functions_required.append("""threshold={threshold}""".format(
                threshold=self.threshold))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = ReLU(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)

