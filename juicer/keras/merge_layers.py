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

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None

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

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

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


class Average(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

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


class Concatenate(Operation):
    INPUTS_PARAM = 'inputs'
    AXIS_PARAM = 'axis'
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self._axis = parameters.get(self.AXIS_PARAM, None)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None
        self.axis = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._axis is not None:
                self.axis = """axis={axis}""".format(axis=self._axis)
                functions_required.append(self.axis)

            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

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
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self._axes = parameters.get(self.AXES_PARAM, None)
        self._normalize = parameters.get(self.NORMALIZE_PARAM, None)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None
        self.axes = None
        self.normalize = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._axes is not None:
                self.axes = get_int_or_tuple(self._axes)
                functions_required.append("""axes={}""".format(self.axes))

            self.normalize = True if int(self._normalize) == 1 else False
            self.normalize = """normalize={normalize}""".format(
                normalize=self.normalize)
            functions_required.append(self.normalize)

            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = dot(
                {add_functions_required}
            )
            """
        ).format(var_name=self.var_name,
                 add_functions_required=self.add_functions_required)


class Maximum(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

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
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

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


class Multiply(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

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


class Subtract(Operation):
    INPUTS_PARAM = 'inputs'
    KWARGS_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._inputs = parameters.get(self.INPUTS_PARAM, None)
        self._kwargs = parameters.get(self.KWARGS_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.inputs = None
        self.kwargs = None
        self.advanced_options = None

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

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self._inputs is not None:
            self.inputs = re.sub(r"\{|\[|\(|\)|\]|\}|\s+", "", str(self._inputs)
                                 ).split(',')

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
            #self.inputs = '[{}]'.format(', '.join(self.inputs))

        functions_required = []
        if self.inputs:
            functions_required.append("""inputs={}""".format(
                '[{}]'.format(', '.join(self.inputs))))
        else:
            raise ValueError(gettext('Parameter {} requires at least 2.')
                             .format(self.INPUTS_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._kwargs:
                # Format kwargs
                self.kwargs = re.sub(r"^\s+|\s+$", "", self._kwargs)
                self.kwargs = re.sub(r"\s+", " ", self.kwargs)
                self.kwargs = re.sub(r"\s*,\s*", ", ", self.kwargs)
                self.kwargs = re.sub(r"\s*=\s*", "=", self.kwargs)

                args = self.kwargs.split(',')
                args_params = self.kwargs.split('=')
                if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                    functions_required.append("""{kwargs}""".format(
                        kwargs=self.kwargs))
                else:
                    raise ValueError(
                        gettext('Parameter {} is invalid').format(
                            self.KWARGS_PARAM))

        self.add_functions_required = ',\n    '.join(functions_required)

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
