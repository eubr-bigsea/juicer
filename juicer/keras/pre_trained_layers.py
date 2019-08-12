# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


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

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
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
        if self.weights is not None and len(self.weights) > 1:
            self.weights.replace("'", "").replace('"', '')
            self.weights = """weights='{weights}'""" \
                .format(weights=self.weights)
            functions_required.append(self.weights)

        if self.input_tensor is not None and len(self.weights) > 1:
            self.input_tensor = """input_tensor={input_tensor}""" \
                .format(beta_initializer=self.input_tensor)
            functions_required.append(self.input_tensor)

        if self.input_shape is not None and len(self.input_shape) > 1:
            self.input_shape = get_tuple(self.input_shape)
            self.input_shape = """input_shape={input_shape}""" \
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
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = InceptionV3(
                include_top={include_top}{add_functions_required}
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

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': ['from tensorflow.python.keras.applications.vgg16 import VGG16']}

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
        if self.weights is not None and len(self.weights) > 1:
            self.weights.replace("'", "").replace('"', '')
            self.weights = """weights='{weights}'""" \
                .format(weights=self.weights)
            functions_required.append(self.weights)

        if self.input_tensor is not None:
            self.input_tensor = """input_tensor={input_tensor}""" \
                .format(beta_initializer=self.input_tensor)
            functions_required.append(self.input_tensor)

        if self.input_shape is not None:
            self.input_shape = get_tuple(self.input_shape)
            self.input_shape = """input_shape={input_shape}""" \
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
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = VGG16(
                include_top={include_top}{add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 include_top=self.include_top,
                 weights=self.weights,
                 classes=self.classes,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)