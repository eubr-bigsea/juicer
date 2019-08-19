# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation
from juicer.util.template_util import *


class Add(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'add',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.advanced_options = True if int(self.advanced_options) == 1 else \
            False

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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

        if self.advanced_options:
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

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'subtract',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = subtract(
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

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'multiply',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

        if self.parents:
            if self.inputs:
                self.inputs = self.inputs + self.parents
            else:
                self.inputs = self.parents

            if len(self.inputs) < 2:
                raise ValueError(
                    gettext('Parameter {} requires at least 2.').format(
                        self.INPUTS_PARAM))
            self.inputs = '{}'.format(', '.join(self.inputs))

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

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'average',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])

        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'maximum',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            tmp_inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = []
            tmp_inputs = tmp_inputs.split(',')
            for tmp in tmp_inputs:
                self.inputs.append('{}'.format(tmp))

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required).replace("'","")


class Minimum(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'minimum',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.axis = parameters.get(self.AXIS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])

        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'concatenate',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self.inputs))
            self.inputs = self.inputs.split(',')

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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

        if self.axis is not None:
            self.axis = """axis={axis}""".format(axis=self.axis)
            functions_required.append(self.axis)

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
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
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

        self.inputs = parameters.get(self.INPUTS_PARAM, None)
        self.axes = parameters.get(self.AXES_PARAM, None)
        self.normalize = parameters.get(self.NORMALIZE_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'dot',
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
        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        for python_code in self.python_code_to_remove:
            self.parents.remove(python_code[0])

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.normalize = True if int(self.normalize) == 1 else False

        for i in range(len(self.parents)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

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
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = dot(
                normalize={normalize}{add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 normalize=self.normalize,
                 add_functions_required=self.add_functions_required)