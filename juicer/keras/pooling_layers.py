# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.util.template_util import *


class AveragePooling1D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'
    KWARG_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._pool_size = parameters.get(self.POOL_SIZE_PARAM, None)
        self._strides = parameters.get(self.STRIDES_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.pool_size = None
        self.strides = None
        self.padding = None
        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.POOL_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'AveragePooling1D',
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
        try:
            self.pool_size = int(self._pool_size)
            functions_required.append("""pool_size={pool_size}""".format(
                pool_size=self.pool_size))
        except:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.POOL_SIZE_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._strides is not None:
                try:
                    self.strides = int(self._strides)
                except:
                    raise ValueError(gettext('Parameter {} is invalid.').format(
                        self.STRIDES_PARAM))
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._padding is not None:
                self.padding = """padding={padding}""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = AveragePooling1D(
                name='{name}'{add_functions_required}
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
    ADVANCED_OPTIONS_PARAM = 'advanced_options'
    KWARG_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._pool_size = parameters.get(self.POOL_SIZE_PARAM, None)
        self._strides = parameters.get(self.STRIDES_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.pool_size = None
        self.strides = None
        self.padding = None
        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.POOL_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'AveragePooling2D',
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
        self.pool_size = get_int_or_tuple(self._pool_size)
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.POOL_SIZE_PARAM))
        else:
            functions_required.append("""pool_size={pool_size}""".format(
                pool_size=self.pool_size))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            self.strides = get_int_or_tuple(self._strides)
            if not self.strides:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.STRIDES_PARAM))
            else:
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._padding is not None:
                self.padding = """padding={padding}""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = AveragePooling2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class AveragePooling3D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'
    KWARG_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._pool_size = parameters.get(self.POOL_SIZE_PARAM, None)
        self._strides = parameters.get(self.STRIDES_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.pool_size = None
        self.strides = None
        self.padding = None
        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self.POOL_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.POOL_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'AveragePooling3D',
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
        self.pool_size = get_int_or_tuple(self._pool_size)
        if not self.pool_size:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.POOL_SIZE_PARAM))
        else:
            functions_required.append("""pool_size={pool_size}""".format(
                pool_size=self.pool_size))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            self.strides = get_int_or_tuple(self._strides)
            if not self.strides:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.STRIDES_PARAM))
            else:
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._padding is not None:
                self.padding = """padding={padding}""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = AveragePooling3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalAveragePooling1D(Operation):
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self.task_name = self.parameters.get('task').get('name')

        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'GlobalAveragePooling1D',
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
            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalAveragePooling1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalAveragePooling2D(Operation):
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self.task_name = self.parameters.get('task').get('name')

        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'GlobalAveragePooling2D',
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
            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalAveragePooling2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalAveragePooling3D(Operation):
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None) or None
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self.task_name = self.parameters.get('task').get('name')

        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'GlobalAveragePooling3D',
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
            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalAveragePooling3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalMaxPooling1D(Operation):
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'GlobalMaxPooling1D',
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
            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalMaxPooling1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalMaxPooling2D(Operation):
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'GlobalMaxPooling2D',
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
            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalMaxPooling2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class GlobalMaxPooling3D(Operation):
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'GlobalMaxPooling3D',
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
            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

            self.add_functions_required = ',\n    '.join(functions_required)
            if self.add_functions_required:
                self.add_functions_required = ',\n    ' + \
                                              self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = GlobalMaxPooling3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class MaxPooling1D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._pool_size = parameters.get(self.POOL_SIZE_PARAM, None)
        self._strides = parameters.get(self.STRIDES_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.pool_size = None
        self.strides = None
        self.padding = None
        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self._pool_size is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.POOL_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'MaxPooling1D',
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
        try:
            self.pool_size = int(self._pool_size)
        except:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.POOL_SIZE_PARAM))
        functions_required.append("""pool_size={pool_size}""".format(
            pool_size=self.pool_size))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False

        if self.advanced_options:
            try:
                self.strides = int(self._strides)
            except:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.STRIDES_PARAM))
            if self.strides is not None:
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))

            if self._padding is not None:
                functions_required.append("""padding='{padding}'""".format(
                    padding=self._padding))

            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = MaxPooling1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class MaxPooling2D(Operation):
    POOL_SIZE_PARAM = 'pool_size'
    STRIDES_PARAM = 'strides'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._pool_size = parameters.get(self.POOL_SIZE_PARAM, None)
        self._strides = parameters.get(self.STRIDES_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.pool_size = None
        self.strides = None
        self.padding = None
        self.data_format = None
        self.kwargs = None
        self.advanced_options = None

        self.task_name = self.parameters.get('task').get('name')

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self._pool_size is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.POOL_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'MaxPooling2D',
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
        self.pool_size = get_int_or_tuple(self._pool_size)
        if self.pool_size:
            functions_required.append("""pool_size={pool_size}""".format(
                pool_size=self.pool_size))
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.POOL_SIZE_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            self.strides = get_int_or_tuple(self._strides)
            if self.strides is not None:
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))
            else:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.STRIDES_PARAM))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = MaxPooling2D(
                name='{name}'{add_functions_required}
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
    KWARG_PARAM = 'kwargs'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'
    TRAINABLE_OPTIONS_PARAM = 'trainable'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._pool_size = parameters.get(self.POOL_SIZE_PARAM, None)
        self._strides = parameters.get(self.STRIDES_PARAM, None)
        self._padding = parameters.get(self.PADDING_PARAM, None)
        self._data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._kwargs = parameters.get(self.KWARG_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)
        self._trainable = parameters.get(self.TRAINABLE_OPTIONS_PARAM, 0)

        self.pool_size = None
        self.strides = None
        self.padding = None
        self.data_format = None
        self.kwargs = None
        self.advanced_options = None
        self.trainable = None

        self.task_name = self.parameters.get('task').get('name')

        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.add_functions_required = ""

        if self._pool_size is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.POOL_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': 'MaxPooling3D',
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
        self.pool_size = get_int_or_tuple(self._pool_size)
        if self.pool_size:
            functions_required.append("""pool_size={pool_size}""".format(
                pool_size=self.pool_size))
        else:
            raise ValueError(gettext('Parameter {} is invalid.').format(
                self.POOL_SIZE_PARAM))

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        self.trainable = True if int(self._trainable) == 1 else \
            False

        if self.advanced_options:
            self.strides = get_tuple(self._strides)
            if self.strides is not None:
                functions_required.append("""strides={strides}""".format(
                    strides=self.strides))
            else:
                raise ValueError(gettext('Parameter {} is invalid.').format(
                    self.STRIDES_PARAM))

            if self._padding is not None:
                self.padding = """padding='{padding}'""".format(
                    padding=self._padding)
                functions_required.append(self.padding)

            if self._data_format is not None:
                self.data_format = """data_format={data_format}""".format(
                    data_format=self._data_format)
                functions_required.append(self.data_format)

            if self._kwargs is not None:
                self.kwargs = self._kwargs.replace(' ', '').split(',')
                functions_required.append(',\n    '.join(self.kwargs))

        if not self.trainable:
            functions_required.append('''trainable=False''')

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + \
                                          self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = MaxPooling3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)
