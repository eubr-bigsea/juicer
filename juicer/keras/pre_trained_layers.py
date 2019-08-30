# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.util.template_util import *


class InceptionV3(Operation):
    INCLUDE_TOP_PARAM = 'include_top'
    WEIGHTS_PARAM = 'weights'
    INPUT_TENSOR_PARAM = 'input_tensor'
    INPUT_SHAPE_PARAM = 'input_shape'
    POOLING_PARAM = 'pooling'
    CLASSES_PARAM = 'classes'
    TRAINABLE_PARAM = 'trainable'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._include_top = parameters.get(self.INCLUDE_TOP_PARAM, None)
        self.weights = parameters.get(self.WEIGHTS_PARAM, None)
        self._input_tensor = parameters.get(self.INPUT_TENSOR_PARAM, None)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._pooling = parameters.get(self.POOLING_PARAM, None)
        self._classes = parameters.get(self.CLASSES_PARAM, None)
        self._trainable = parameters.get(self.TRAINABLE_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.include_top = None
        self.input_tensor = None
        self.input_shape = None
        self.pooling = None
        self.classes = None
        self.trainable = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()

        if self.WEIGHTS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.WEIGHTS_PARAM))

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': ['from keras.applications.inception_v3 '
                                       'import InceptionV3']}

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

        self.trainable = True if int(self._trainable) == 1 else False

        self.include_top = True if int(self._include_top) == 1 else False
        functions_required = ["""include_top={include_top}""".format(
            include_top=self.include_top)]

        if self.weights is not None and self.weights.strip():
            self.weights = self.weights.replace("'", "").replace('"', '')
            self.weights = """weights='{weights}'""".format(
                weights=self.weights)
            functions_required.append(self.weights)

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._input_tensor:
                self.input_tensor = get_tuple(self._input_tensor)
                if not self.input_tensor:
                    raise ValueError(gettext('Parameter {} is invalid').format(
                        self.INPUT_TENSOR_PARAM))
                else:
                    if self.input_tensor is not None and len(
                            self.input_tensor) > 1:
                        functions_required.append("""input_tensor={}""".format(
                            self.input_tensor))

            if not self.include_top:
                self.input_shape = get_tuple(self._input_shape)
                if self.input_shape is not None and len(
                        self.input_shape) > 1:
                    functions_required.append("""input_shape={}""".format(
                        self.input_shape))

                if self._pooling is not None:
                    self.pooling = """pooling='{}'""".format(self.pooling)
                    functions_required.append(self.pooling)
            else:
                self.input_shape = None
                self.pooling = None
                if self.weights is None:
                    if self._classes is not None:
                        self.classes = """classes={}""".format(
                            self._classes)
                        functions_required.append(self.classes)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = InceptionV3(
                {add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class VGG16(Operation):
    INCLUDE_TOP_PARAM = 'include_top'
    WEIGHTS_PARAM = 'weights'
    INPUT_TENSOR_PARAM = 'input_tensor'
    INPUT_SHAPE_PARAM = 'input_shape'
    POOLING_PARAM = 'pooling'
    CLASSES_PARAM = 'classes'
    TRAINABLE_PARAM = 'trainable'
    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._include_top = parameters.get(self.INCLUDE_TOP_PARAM, None)
        self.weights = parameters.get(self.WEIGHTS_PARAM, None)
        self._input_tensor = parameters.get(self.INPUT_TENSOR_PARAM, None)
        self._input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self._pooling = parameters.get(self.POOLING_PARAM, None)
        self._classes = parameters.get(self.CLASSES_PARAM, None)
        self._trainable = parameters.get(self.TRAINABLE_PARAM, None)
        self._advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM, 0)

        self.include_top = None
        self.input_tensor = None
        self.input_shape = None
        self.pooling = None
        self.classes = None
        self.trainable = None
        self.advanced_options = None

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()

        if self.WEIGHTS_PARAM is None:
            raise ValueError(
                gettext('Parameter {} is required').format(self.WEIGHTS_PARAM))

        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': ['from keras.applications.VGG16 '
                                       'import VGG16']}

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

        self.trainable = True if int(self._trainable) == 1 else False

        self.include_top = True if int(self._include_top) == 1 else False
        functions_required = ["""include_top={include_top}""".format(
            include_top=self.include_top)]

        if self.weights is not None and self.weights.strip():
            self.weights = self.weights.replace("'", "").replace('"', '')
            self.weights = """weights='{weights}'""".format(
                weights=self.weights)
            functions_required.append(self.weights)

        self.advanced_options = True if int(self._advanced_options) == 1 else \
            False
        if self.advanced_options:
            if self._input_tensor:
                self.input_tensor = get_tuple(self._input_tensor)
                if not self.input_tensor:
                    raise ValueError(gettext('Parameter {} is invalid').format(
                        self.INPUT_TENSOR_PARAM))
                else:
                    if self.input_tensor is not None and len(
                            self.input_tensor) > 1:
                        functions_required.append("""input_tensor={}""".format(
                            self.input_tensor))

            if not self.include_top:
                self.input_shape = get_tuple(self._input_shape)
                if self.input_shape is not None and len(
                        self.input_shape) > 1:
                    functions_required.append("""input_shape={}""".format(
                        self.input_shape))

                if self._pooling is not None:
                    self.pooling = """pooling='{}'""".format(self.pooling)
                    functions_required.append(self.pooling)
            else:
                self.input_shape = None
                self.pooling = None
                if self.weights is None:
                    if self._classes is not None:
                        self.classes = """classes={}""".format(
                            self._classes)
                        functions_required.append(self.classes)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        return dedent(
            """
            {var_name} = VGG16(
                {add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)
