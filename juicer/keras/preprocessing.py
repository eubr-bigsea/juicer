# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


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
    INTERPOLATION_PARAM = 'interpolation'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.featurewise_center = parameters.get(
            self.FEATUREWISE_CENTER_PARAM, None)
        self.samplewise_center = parameters.get(
            self.SAMPLEWISE_CENTER_PARAM, None)
        self.featurewise_std_normalization = parameters.get(
            self.FEATUREWISE_STD_NORMALIZATION_PARAM, None)
        self.samplewise_std_normalization = parameters.get(
            self.SAMPLEWISE_STD_NORMALIZATION_PARAM, None)
        self.zca_epsilon = parameters.get(self.ZCA_EPSILON_PARAM, None)
        self.zca_whitening = parameters.get(self.ZCA_WHITENING_PARAM, None)
        self.rotation_range = parameters.get(self.ROTATION_RANGE_PARAM, None)
        self.width_shift_range = parameters.get(
            self.WIDTH_SHIFT_RANGE_PARAM, None)
        self.height_shift_range = parameters.get(
            self.HEIGHT_SHIFT_RANGE_PARAM, None)
        self.brightness_range = parameters.get(
            self.BRIGHTNESS_RANGE_PARAM, None)
        self.shear_range = parameters.get(self.SHEAR_RANGE_PARAM, None)
        self.zoom_range = parameters.get(self.ZOOM_RANGE_PARAM, None)
        self.channel_shift_range = parameters.get(
            self.CHANNEL_SHIFT_RANGE_PARAM, None)
        self.fill_mode = parameters.get(self.FILL_MODE_PARAM, None)
        self.cval = parameters.get(self.CVAL_PARAM, None)
        self.horizontal_flip = parameters.get(self.HORIZONTAL_FLIP_PARAM, None)
        self.vertical_flip = parameters.get(self.VERTICAL_FLIP_PARAM, None)
        self.rescale = parameters.get(self.RESCALE_PARAM, None)
        self.preprocessing_function = parameters.get(
            self.PREPROCESSING_FUNCTION_PARAM, None)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self._validation_split = parameters.get(
            self.VALIDATION_SPLIT_PARAM, 0.0)
        self.dtype = parameters.get(self.DTYPE_PARAM, None)

        self.target_size = parameters.get(self.TARGET_SIZE_PARAM, None)
        self.color_mode = parameters.get(self.COLOR_MODE_PARAM, None)
        self.class_mode = parameters.get(self.CLASS_MODE_PARAM, None)
        self.batch_size = parameters.get(self.BATCH_SIZE_PARAM, None)
        self.shuffle = parameters.get(self.SHUFFLE_PARAM, None)
        self.interpolation = parameters.get(self.INTERPOLATION_PARAM, None)

        self.image_train = None
        self.image_validation = None
        self.image_test = None
        self.validation_split = None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True
        self.add_functions_required = ''
        self.add_functions_required_flow_from_directory = ''

        if self.TARGET_SIZE_PARAM not in parameters or \
                self.TARGET_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(
                self.TARGET_SIZE_PARAM))

        if self.BATCH_SIZE_PARAM not in parameters or \
                self.BATCH_SIZE_PARAM is None:
            raise ValueError(gettext('Parameter {} is required').format(
                self.BATCH_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': 'ImageDataGenerator',
                            'others': None}

    def treatment(self):
        parents_by_port = self.parameters.get('parents_by_port', [])
        if len(parents_by_port) == 1:
            if str(parents_by_port[0][0]) == 'train-image':
                self.image_train = parents_by_port[0]
                self.image_validation = None
                self.image_test = None
            elif str(parents_by_port[0][0]) == 'validation-image':
                self.image_train = None
                self.image_validation = parents_by_port[0]
                self.image_test = None
            elif str(parents_by_port[0][0]) == 'test-image':
                self.image_train = None
                self.image_validation = None
                self.image_test = parents_by_port[0]

        if not (self.image_train or self.image_validation or self.image_test):
            raise ValueError(gettext('You need to correctly specify the '
                                     'ports for training or validation or test.'
                                     ))

        if self.image_train:
            self.image_train = convert_variable_name(
                self.image_train[1]) + '_' + \
                               convert_variable_name(self.image_train[0])

        if self.image_validation:
            self.image_validation = convert_variable_name(
                self.image_validation[1]) + '_' + \
                               convert_variable_name(self.image_validation[0])

        if self.image_test:
            self.image_test = convert_variable_name(
                self.image_test[1]) + '_' + \
                               convert_variable_name(self.image_test[0])

        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        functions_required_flow_from_directory = []
        if self.image_train:
            functions_required_flow_from_directory.append(
                """directory={image_train}""".format(
                    image_train=self.image_train))

        if self.image_validation:
            functions_required_flow_from_directory.append(
                """directory={image_validation}""".format(
                    image_validation=self.image_validation))

        if self.image_test:
            functions_required_flow_from_directory.append(
                """directory={image_test}""".format(
                    image_test=self.image_test))

        if self.target_size:
            target_size = get_int_or_tuple(self.target_size)
            if target_size:
                functions_required_flow_from_directory.append(
                    """target_size={target_size}""".format(
                        target_size=target_size))
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.TARGET_SIZE_PARAM))

        if self.color_mode:
            functions_required_flow_from_directory.append(
                """color_mode='{color_mode}'""".format(
                    color_mode=self.color_mode))

        if self.class_mode:
            functions_required_flow_from_directory.append(
                """class_mode='{class_mode}'""".format(
                    class_mode=self.class_mode))

        if self.batch_size:
            functions_required_flow_from_directory.append(
                """batch_size={batch_size}""".format(
                    batch_size=self.batch_size))

        if self.interpolation:
            functions_required_flow_from_directory.append(
                """interpolation='{interpolation}'""".format(
                    interpolation=self.interpolation))

        self.add_functions_required_flow_from_directory = ',\n    '.join(
            functions_required_flow_from_directory)

        functions_required = []

        if self.zca_epsilon is not None:
            try:
                functions_required.append(
                    """zca_epsilon={zca_epsilon}""".format(
                        zca_epsilon=float(self.zca_epsilon)))
            except:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.ZCA_EPSILON_PARAM))

        featurewise_center = True \
            if int(self.featurewise_center) == 1 else False
        functions_required.append(
            """featurewise_center={featurewise_center}""".format(
                featurewise_center=featurewise_center))

        samplewise_center = True \
            if int(self.samplewise_center) == 1 else False
        functions_required.append(
            """samplewise_center={samplewise_center}""".format(
                samplewise_center=samplewise_center))

        featurewise_std_normalization = True \
            if int(self.featurewise_std_normalization) == 1 else False
        functions_required.append(
            """featurewise_std_normalization={featurewise_std_normalization}"""
                .format(
                featurewise_std_normalization=featurewise_std_normalization))

        samplewise_std_normalization = True \
            if int(self.samplewise_std_normalization) == 1 else False
        functions_required.append(
            """samplewise_std_normalization={samplewise_std_normalization}""".
                format(
                samplewise_std_normalization=samplewise_std_normalization))

        zca_whitening = True if int(self.zca_whitening) == 1 else False
        functions_required.append("""zca_whitening={zca_whitening}""".format(
            zca_whitening=zca_whitening))

        try:
            functions_required.append(
                """rotation_range={rotation_range}""".format(
                    rotation_range=int(self.rotation_range)))
        except:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.ROTATION_RANGE_PARAM))

        width_shift_range = string_to_int_float_list(self.width_shift_range)
        if width_shift_range is not None:
            functions_required.append(
                """width_shift_range={width_shift_range}""".format(
                    width_shift_range=width_shift_range))
        else:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.WIDTH_SHIFT_RANGE_PARAM))

        height_shift_range = string_to_int_float_list(self.height_shift_range)
        if height_shift_range is not None:
            functions_required.append(
                """height_shift_range={height_shift_range}""".format(
                    height_shift_range=height_shift_range))
        else:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.HEIGHT_SHIFT_RANGE_PARAM))

        if self.brightness_range is not None:
            brightness_range = string_to_list(self.brightness_range)
            if brightness_range is not None and len(brightness_range) == 2:
                functions_required.append(
                    """brightness_range={brightness_range}""".format(
                        brightness_range=self.brightness_range))
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.BRIGHTNESS_RANGE_PARAM))

        try:
            shear_range = float(self.shear_range)
        except:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.SHEAR_RANGE_PARAM))
        functions_required.append("""shear_range={shear_range}""".format(
            shear_range=shear_range))

        zoom_range = string_to_list(self.zoom_range)
        if zoom_range and len(zoom_range) <= 2:
            if len(zoom_range) == 1:
                functions_required.append("""zoom_range={zoom_range}""".format(
                    zoom_range=zoom_range[0]))
            else:
                functions_required.append("""zoom_range={zoom_range}""".format(
                    zoom_range=zoom_range))
        else:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.ZOOM_RANGE_PARAM))

        try:
            functions_required.append(
                """channel_shift_range={channel_shift_range}""".format(
                    channel_shift_range=float(
                self.channel_shift_range)))
        except:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.CHANNEL_SHIFT_RANGE_PARAM))

        if self.fill_mode:
            functions_required.append("""fill_mode='{fill_mode}'""".format(
                fill_mode=self.fill_mode))

        if self.fill_mode == 'constant':
            try:
                functions_required.append("""cval={cval}""".format(
                    cval=float(self.cval)))
            except:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.CVAL_PARAM))

        horizontal_flip = True if int(self.horizontal_flip) == 1 else False
        functions_required.append(
            """horizontal_flip={horizontal_flip}""".format(
                horizontal_flip=horizontal_flip))

        vertical_flip = True if int(self.vertical_flip) == 1 else False
        functions_required.append("""vertical_flip={vertical_flip}""".format(
            vertical_flip=vertical_flip))

        if self.rescale is not None:
            _rescale = rescale(self.rescale)
            if _rescale is not None:
                functions_required.append("""rescale={rescale}""".format(
                    rescale=_rescale))
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.RESCALE_PARAM))

        '''TO_DO - ADD preprocessing_function IN THE FUTURE'''

        if self.data_format:
            functions_required.append("""data_format={data_format}""".format(
                data_format=self.data_format))

        # In case of the operation is creating the image data
        self.validation_split = float(self._validation_split)
        if self.image_train:
            if self.validation_split:
                if self.validation_split > 0:
                    functions_required.append(
                        """validation_split={validation_split}""".format(
                            validation_split=abs(float(self.validation_split))))

        if self.dtype is not None:
            functions_required.append("""dtype='{dtype}'""".format(
                dtype=self.dtype))

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        if self.image_train:
            if self.validation_split > 0:
                return dedent(
                    """
                    {var_name}_datagen = ImageDataGenerator(
                        {add_functions_required}
                    )
                    train_{var_name} = {var_name}_datagen.flow_from_directory(
                        {add_functions_required_flow_from_directory},
                        subset='training'
                    )
                    validation_{var_name} = {var_name}_datagen.flow_from_directory(
                        {add_functions_required_flow_from_directory},
                        subset='validation'
                    )
                    
                    """
                ).format(var_name=self.var_name,
                         add_functions_required=self.add_functions_required,
                         add_functions_required_flow_from_directory=self.
                         add_functions_required_flow_from_directory)
            else:
                return dedent(
                    """
                    {var_name}_datagen = ImageDataGenerator(
                        {add_functions_required}
                    )
                    train_{var_name} = {var_name}_datagen.flow_from_directory(
                        {add_functions_required_flow_from_directory}
                    )
                    validation_{var_name} = None
                    
                    """
                ).format(var_name=self.var_name,
                         add_functions_required=self.add_functions_required,
                         add_functions_required_flow_from_directory=self.
                         add_functions_required_flow_from_directory)

        if self.image_validation:
            return dedent(
                """
                {var_name}_datagen = ImageDataGenerator(
                    {add_functions_required}
                )
                validation_{var_name} = {var_name}_datagen.flow_from_directory(
                    {add_functions_required_flow_from_directory}
                )
                
                """
            ).format(var_name=self.var_name,
                     add_functions_required=self.add_functions_required,
                     add_functions_required_flow_from_directory=self.
                     add_functions_required_flow_from_directory)

        if self.image_test:
            return dedent(
                """
                {var_name}_datagen = ImageDataGenerator(
                    {add_functions_required}
                )
                test_{var_name} = {var_name}_datagen.flow_from_directory(
                    {add_functions_required_flow_from_directory}
                )
                
                """
            ).format(var_name=self.var_name,
                     add_functions_required=self.add_functions_required,
                     add_functions_required_flow_from_directory=self.
                     add_functions_required_flow_from_directory)


class SequenceGenerator(Operation):
    BATCH_SIZE_PARAM = 'batch_size'
    VALIDATION_SPLIT_PARAM = 'validation_split'
    SHUFFLE_PARAM = 'shuffle'
    SHAPE_PARAM = 'shape'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.batch_size = int(parameters.get(self.BATCH_SIZE_PARAM, None))
        self.validation_split = float(parameters.get(
            self.VALIDATION_SPLIT_PARAM, 0))
        self._shuffle = int(parameters.get(self.SHUFFLE_PARAM, None))
        self._shape = parameters.get(self.SHAPE_PARAM, None)

        self.shuffle = True
        self.shape = None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        self.sequence_training = None
        self.sequence_validation = None
        self.sequence_test = None
        self.shuffle = None

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.has_external_code = True

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': ['from os import walk, listdir',
                                       'from os.path import isfile, join',
                                       'from keras.utils '
                                       'import to_categorical']
                            }

    def treatment(self):
        parents_by_port = self.parameters.get('parents_by_port', [])

        if len(parents_by_port) == 1:
            if str(parents_by_port[0][0]) == 'train-sequence':
                self.sequence_training = parents_by_port[0]
                self.sequence_validation = None
                self.sequence_test = None
            elif str(parents_by_port[0][0]) == 'validation-sequence':
                self.sequence_training = None
                self.sequence_test = None
                self.sequence_validation = parents_by_port[0]
            elif str(parents_by_port[0][0]) == 'test-sequence':
                self.sequence_training = None
                self.sequence_validation = None
                self.sequence_test = parents_by_port[0]

        if not (self.sequence_training or self.sequence_validation or
                self.sequence_test):
            raise ValueError(gettext('You need to correctly specify the '
                                     'ports for training or validation or '
                                     'test.'
                                     )
                             )

        if self.sequence_training and self.sequence_validation and \
                self.validation_split:
            raise ValueError(gettext('Is impossible to use validation split '
                                     'option > 0 and video reader for training '
                                     'and validation data.'))

        if self.sequence_training:
            self.sequence_training = convert_variable_name(
                self.sequence_training[1]) + '_' + convert_variable_name(
                self.sequence_training[0])

        if self.sequence_validation:
            self.sequence_validation = convert_variable_name(
                self.sequence_validation[1]) + '_' + convert_variable_name(
                self.sequence_validation[0])

        if self.sequence_test:
            self.sequence_test = convert_variable_name(
                self.sequence_test[1]) + '_' + convert_variable_name(
                self.sequence_test[0])

        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.shuffle = True if int(self._shuffle) == 1 else False

        if self.batch_size < 1:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.BATCH_SIZE_PARAM))

        if self.validation_split < 0 or self.validation_split > 1:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.VALIDATION_SPLIT_PARAM))

        self.shape = get_tuple(self._shape)

    def external_code(self):
        if self.sequence_training:
            return dedent(
                """     
                class SequenceGenerator(keras.utils.Sequence):
                    def __init__(self, sequence_path=[],
                                       batch_size=16,
                                       shape=None,
                                       n_classes=0, 
                                       shuffle=True):
                                       
                        self.sequence_path = sequence_path
                        self.batch_size = batch_size
                        self.shape = shape
                        self.n_classes = n_classes
                        self.shuffle = shuffle
                        self.classes = []
                        self.to_fit = True
                        
                        self.on_epoch_end()
                        
                    def on_epoch_end(self):
                        self.instances_indexes = np.arange(len(self.sequence_path))
                        if self.shuffle:
                            np.random.shuffle(self.instances_indexes)
                    
                    def to_np_array(self, data):
                        import cv2
                        
                        img = cv2.imread(data, cv2.IMREAD_UNCHANGED)
                        resized = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
                        
                        return resized

                            
                    def __len__(self):
                        return int(len(self.sequence_path) / self.batch_size)
                        
                    def __getitem__(self, index):
                        # Generate indexes of the batch
                        indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]
                        
                        # Find list of IDs
                        batch_ids = [self.sequence_path[k] for k in indexes]
                        
                        # Generate data
                        X, y = self.__data_generation(batch_ids)
                        
                        if self.to_fit:
                            return X, y
                        else:
                            return X
                    
                    def __data_generation(self, batch_ids):
                        data = np.empty((self.batch_size, *self.shape))
                        classes = np.empty(self.batch_size, dtype=int)
                        
                        for i, instance_id in enumerate(batch_ids):
                            try:
                                x = np.load(instance_id[0])
                            except:
                                x = self.to_np_array(instance_id[0])
                                
                            data[i,] = x
                            classes[i] = class_mapping[instance_id[1]]
                        
                        self.classes += list(classes)
                        
                        return (
                            data,
                            to_categorical(
                                classes,
                                num_classes=self.n_classes
                            )
                        )
                
                {class_mapping}
            
                def sequence_path_reader(path='', validation_split=0.0):
                    if path:
                        classes = [x[0] for x in walk(path)][1:]
                    else:
                        raise ValueError(gettext('Data set path is invalid.'))
                    
                    if classes:
                        files = []
                        id = 0
                        for class_path in classes:
                            cls = class_path.split('/')[-1] # Get only class name
                            if not cls in class_mapping:
                                class_mapping[cls] = id
                                id += 1
                
                            files += [(class_path+'/'+f, cls) for f in listdir(class_path) if isfile(join(class_path, f))]
                            
                        if validation_split:
                            _index = int(len(files)*validation_split)
                            
                            training_files = files[_index:]
                            validation_files = files[0:_index]
                            
                            return {_return}
                        else:
                            return files
                            
                """
            ).format(_return='{"training": training_files, '
                             '"validation": validation_files}',
                     class_mapping='class_mapping = {}')
        else:
            return dedent("""""")

    def generate_code(self):
        if self.sequence_training:
            if self.validation_split == 0:
                return dedent(
                    """
                    training_sequence_path = sequence_path_reader(
                        path={path}
                    )
                    
                    train_{var_name} = SequenceGenerator(
                        sequence_path=training_sequence_path,
                        batch_size={batch_size},
                        shape={shape},
                        n_classes=len(class_mapping),
                        shuffle={shuffle}
                    )
                    
                    predict_train_{var_name} = SequenceGenerator(
                        sequence_path=train_{var_name}.sequence_path,
                        batch_size={batch_size},
                        shape={shape},
                        n_classes=len(class_mapping),
                        shuffle=False
                    )

                    """
                ).format(var_name=self.var_name,
                         path=self.sequence_training,
                         batch_size=self.batch_size,
                         shuffle=self.shuffle,
                         shape=self.shape)
            else:
                return dedent(
                    """
                    sequence_path = sequence_path_reader(
                        path={path},
                        validation_split={validation_split}
                    )
                    
                    train_{var_name} = SequenceGenerator(
                        sequence_path=sequence_path[{subset_training}],
                        batch_size={batch_size},
                        shape={shape},
                        n_classes=len(class_mapping),
                        shuffle={shuffle}
                    )
                    
                    predict_train_{var_name} = SequenceGenerator(
                        sequence_path=train_{var_name}.sequence_path[{subset_training}],
                        batch_size={batch_size},
                        shape={shape},
                        n_classes=len(class_mapping),
                        shuffle=False
                    )

                    validation_{var_name} = SequenceGenerator(
                        sequence_path=sequence_path[{subset_validation}],
                        batch_size={batch_size},
                        shape={shape},
                        n_classes=len(class_mapping),
                        shuffle={shuffle}
                    )
                    
                    predict_validation_{var_name} = SequenceGenerator(
                        sequence_path=validation_{var_name}.sequence_path[{subset_validation}],
                        batch_size={batch_size},
                        shape={shape},
                        n_classes=len(class_mapping),
                        shuffle=False
                    )
                    """
                ).format(var_name=self.var_name,
                         path=self.sequence_training,
                         validation_split=self.validation_split,
                         batch_size=self.batch_size,
                         shuffle=self.shuffle,
                         shape=self.shape)

        if self.sequence_validation:
            return dedent(
                """
                validation_sequence_path = sequence_path_reader(
                        path={path}
                )
                    
                validation_{var_name} = SequenceGenerator(
                    sequence_path=validation_sequence_path,
                    batch_size={batch_size},
                    shape={shape},
                    n_classes=len(class_mapping),
                    shuffle={shuffle}
                )
                
                predict_validation_{var_name} = SequenceGenerator(
                    sequence_path=validation_{var_name}.sequence_path,
                    batch_size={batch_size},
                    shape={shape},
                    n_classes=len(class_mapping),
                    shuffle=False
                )
                """
            ).format(var_name=self.var_name,
                     path=self.sequence_validation,
                     batch_size=self.batch_size,
                     shuffle=self.shuffle,
                     shape=self.shape)

        if self.sequence_test:
            return dedent(
                """
                test_sequence_path = sequence_path_reader(
                        path={path}
                )
                    
                test_{var_name} = SequenceGenerator(
                    sequence_path=test_sequence_path,
                    batch_size={batch_size},
                    shape={shape},
                    n_classes=len(class_mapping),
                    shuffle={shuffle}
                )
                """
            ).format(var_name=self.var_name,
                     path=self.sequence_test,
                     batch_size=self.batch_size,
                     shuffle=self.shuffle,
                     shape=self.shape)


class VideoGenerator(Operation):
    DIMENSIONS_PARAM = 'dimensions'
    CHANNELS_PARAM = 'channels'
    CROPPING_STRATEGY_PARAM = 'cropping_strategy'
    BATCH_SIZE_PARAM = 'batch_size'
    SHUFFLE_PARAM = 'shuffle'
    VALIDATION_SPLIT_PARAM = 'validation_split'
    CROPPING_STRATEGY_PARAM = 'cropping_strategy'
    RANDOM_FRAMES_PARAM = 'random_frames'
    RANDOM_HEIGHT_PARAM = 'random_height'
    RANDOM_WIDTH_PARAM = 'random_width'
    RANDOM_CHANNEL_PARAM = 'random_channel'
    VIDEO_FRAMES_PARAM = 'video_frames'
    VIDEO_HEIGHT_PARAM = 'video_height'
    VIDEO_WIDTH_PARAM = 'video_width'
    VIDEO_CHANNEL_PARAM = 'video_channel'
    APPLY_TRANSFORMATIONS_PARAM = 'apply_transformations'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self._dimensions = parameters.get(self.DIMENSIONS_PARAM, None)
        self._channels = parameters.get(self.CHANNELS_PARAM, None)
        self._cropping_strategy = parameters.get(self.CROPPING_STRATEGY_PARAM,
                                                None)
        self._batch_size = parameters.get(self.BATCH_SIZE_PARAM, None)
        self._shuffle = parameters.get(self.SHUFFLE_PARAM, None)
        self._validation_split = parameters.get(self.VALIDATION_SPLIT_PARAM,
                                               None)

        self.cropping_strategy = parameters.get(self.CROPPING_STRATEGY_PARAM,
                                                None)
        self._random_frames = parameters.get(self.RANDOM_FRAMES_PARAM, None)
        self._random_height = parameters.get(self.RANDOM_HEIGHT_PARAM, None)
        self._random_width = parameters.get(self.RANDOM_WIDTH_PARAM, None)
        self._random_channel = parameters.get(self.RANDOM_CHANNEL_PARAM, None)
        self._frames = parameters.get(self.VIDEO_FRAMES_PARAM, None)
        self._height = parameters.get(self.VIDEO_HEIGHT_PARAM, None)
        self._width = parameters.get(self.VIDEO_WIDTH_PARAM, None)
        self._channel = parameters.get(self.VIDEO_CHANNEL_PARAM, None)
        self._apply_transformations = parameters.get(
            self.APPLY_TRANSFORMATIONS_PARAM, None)

        self.batch_size = None
        self.validation_split = None
        self.channels = None

        self.video_training = None
        self.video_validation = None
        self.video_test = None
        self.shuffle = None
        self.dimensions = None
        self.random_frames = None
        self.random_height = None
        self.random_width = None
        self.random_channel = None
        self.frames = None
        self.height = None
        self.width = None
        self.channel = None
        self.apply_transformations = None

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True

        if self.DIMENSIONS_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(
                self.DIMENSIONS_PARAM))
        if self.CHANNELS_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(
                self.CHANNELS_PARAM))
        if self.BATCH_SIZE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(
                self.BATCH_SIZE_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.has_external_code = True

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': ['from os import walk, listdir',
                                       'from os.path import isfile, join',
                                       'from keras.utils '
                                       'import to_categorical']
                            }

    def treatment(self):
        parents_by_port = self.parameters.get('parents_by_port', [])

        if len(parents_by_port) == 1:
            if str(parents_by_port[0][0]) == 'train-video':
                self.video_training = parents_by_port[0]
                self.video_validation = None
                self.video_test = None
            elif str(parents_by_port[0][0]) == 'validation-video':
                self.video_training = None
                self.video_test = None
                self.video_validation = parents_by_port[0]
            elif str(parents_by_port[0][0]) == 'test-video':
                self.video_training = None
                self.video_validation = None
                self.video_test = parents_by_port[0]

        if not (self.video_training or self.video_validation or
                self.video_test):
            raise ValueError(gettext('You need to correctly specify the '
                                     'ports for training or validation or '
                                     'test.'
                                     )
                             )

        if self.video_training and self.video_validation and \
                self.validation_split:
            raise ValueError(gettext('Is impossible to use validation split '
                                     'option > 0 and video reader for training '
                                     'and validation data.'))

        if self.video_training:
            self.video_training = convert_variable_name(self.video_training[1])\
                                  + '_' + \
                                  convert_variable_name(self.video_training[0])

        if self.video_validation:
            self.video_validation = convert_variable_name(
                self.video_validation[1]) + '_' + convert_variable_name(
                self.video_validation[0])

        if self.video_test:
            self.video_test = convert_variable_name(
                self.video_test[1]) + '_' + convert_variable_name(
                self.video_test[0])

        self.parents = convert_parents_to_variable_name(self.parameters
                                                        .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        self.shuffle = True if int(self._shuffle) == 1 else False

        self.dimensions = get_tuple(self._dimensions)
        if self.dimensions is None:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.DIMENSIONS_PARAM))

        self.channels = int(self._channels)
        if self.channels < 0:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.VIDEO_CHANNELS_PARAM))

        self.batch_size = int(self._batch_size)
        if self.batch_size < 1:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.BATCH_SIZE_PARAM))

        self.validation_split = float(self._validation_split)
        if self.validation_split < 0:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.VALIDATION_SPLIT_PARAM))

        self.apply_transformations = True \
            if int(self._apply_transformations) == 1 else False

        if self.apply_transformations:
            if self.cropping_strategy == 'random':
                if self._random_frames is not None:
                    self.random_frames = get_interval(self._random_frames)
                    if not self.random_frames:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.RANDOM_FRAMES_PARAM))
                else:
                    self.random_frames = ':'

                self.random_height = get_random_interval(self._random_height)
                if not self.random_height:
                    raise ValueError(gettext('Parameter {} is invalid.')
                                     .format(self.RANDOM_HEIGHT_PARAM))

                self.random_width = get_random_interval(self._random_width)
                if not self.random_width:
                    raise ValueError(gettext('Parameter {} is invalid.')
                                     .format(self.RANDOM_WIDTH_PARAM))

                if self._random_channel is not None:
                    self.random_channel = get_interval(self._random_channel)
                    if not self.random_channel:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.RANDOM_CHANNEL_PARAM))
                else:
                    self.random_channel = ':'

            elif self.cropping_strategy == 'center':
                if self._frames is not None:
                    self.frames = get_interval(self._frames)
                    if not self.frames:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.VIDEO_FRAMES_PARAM))
                else:
                    self.frames = ':'

                if self._height is not None:
                    self.height = get_interval(self._height)
                    if not self.height:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.VIDEO_HEIGHT_PARAM))
                else:
                    self.height = ':'

                if self._width is not None:
                    self.width = get_interval(self._width)
                    if not self.width:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.VIDEO_WIDTH_PARAM))
                else:
                    self.width = ':'

                if self._channel is not None:
                    self.channel = get_interval(self._channel)
                    if not self.channel:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.CHANNEL_PARAM))
                else:
                    self.channel = ':'

            elif self.cropping_strategy is None:
                pass
        else:
            self.cropping_strategy = None

    def external_code(self):
        generator_name = ''

        if self.video_training:
            if self.validation_split == 0:
                generator_name = 'VideoGeneratorTraining'
            else:
                generator_name = 'VideoGenerator'
        if self.video_validation:
            generator_name = 'VideoGeneratorValidation'

        if self.video_test:
            generator_name = 'VideoGeneratorTest'

        if self.cropping_strategy is None:
            if self.video_training:
                return dedent(
                    """     
                    class {generator_name}(keras.utils.Sequence):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            self.to_fit = True
                            
                            self.on_epoch_end()
                            
                        def on_epoch_end(self):
                            self.instances_indexes = np.arange(len(self.videos_path))
                            if self.shuffle:
                                np.random.shuffle(self.instances_indexes)
                                
                        def __len__(self):
                            return int(len(self.videos_path) / self.batch_size)
                            
                        def __getitem__(self, index):
                            # Generate indexes of the batch
                            indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]
                            
                            # Find list of IDs
                            batch_ids = [self.videos_path[k] for k in indexes]
                            
                            # Generate data
                            X, y = self.__data_generation(batch_ids)
                            
                            if self.to_fit:
                                return X, y
                            else:
                                return X
                        
                        def __data_generation(self, batch_ids):
                            data = np.empty(self.data_shape)
                            classes = np.empty(self.data_shape[0], dtype=int)
                            
                            for i, instance_id in enumerate(batch_ids):
                                x = np.load(instance_id[0])['frames']
                                data[i,] = x
                                classes[i] = class_mapping[instance_id[1]]
                            
                            self.classes += list(classes)
                            
                            return (
                                data,
                                to_categorical(
                                    classes,
                                    num_classes=self.n_classes
                                )
                            )
                    
                    {class_mapping}
                
                    def video_path_reader(path='', validation_split=0.0):
                        if path:
                            classes = [x[0] for x in walk(path)][1:]
                        else:
                            raise ValueError(gettext('Data set path is invalid.'))
                        
                        if classes:
                            files = []
                            id = 0
                            for class_path in classes:
                                cls = class_path.split('/')[-1] # Get only class name
                                if not cls in class_mapping:
                                    class_mapping[cls] = id
                                    id += 1
                    
                                files += [(class_path+'/'+f, cls) for f in listdir(class_path) if isfile(join(class_path, f))]
                                
                            if validation_split:
                                _index = int(len(files)*validation_split)
                                
                                training_files = files[_index:]
                                validation_files = files[0:_index]
                                
                                return {_return}
                            else:
                                return files
                                
                    """
                ).format(generator_name=generator_name,
                         _return='{"training": training_files, '
                                 '"validation": validation_files}',
                         class_mapping='class_mapping = {}')

            elif self.video_validation:
                return dedent(
                    """     
                    class {generator_name}(keras.utils.Sequence):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            self.to_fit = True
                            
                            self.on_epoch_end()
                            
                        def on_epoch_end(self):
                            self.instances_indexes = np.arange(len(self.videos_path))
                            if self.shuffle:
                                np.random.shuffle(self.instances_indexes)
                                
                        def __len__(self):
                            return int(len(self.videos_path) / self.batch_size)
                            
                        def __getitem__(self, index):
                            # Generate indexes of the batch
                            indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]
                            
                            # Find list of IDs
                            batch_ids = [self.videos_path[k] for k in indexes]
                            
                            # Generate data
                            X, y = self.__data_generation(batch_ids)
                            
                            if self.to_fit:
                                return X, y
                            else:
                                return X
                        
                        def __data_generation(self, batch_ids):
                            data = np.empty(self.data_shape)
                            classes = np.empty(self.data_shape[0], dtype=int)
                            
                            for i, instance_id in enumerate(batch_ids):
                                x = np.load(instance_id[0])['frames']
                                data[i,] = x
                                classes[i] = class_mapping[instance_id[1]]
                            
                            self.classes += list(classes)
                            
                            return (
                                data,
                                to_categorical(
                                    classes,
                                    num_classes=self.n_classes
                                )
                            )
                    """
                ).format(generator_name=generator_name)
            elif self.video_test:
                return dedent(
                    """     
                    class {generator_name}(keras.utils.Sequence):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            self.to_fit = False
                            
                            self.on_epoch_end()
                            
                        def on_epoch_end(self):
                            self.instances_indexes = np.arange(len(self.videos_path))
                            if self.shuffle:
                                np.random.shuffle(self.instances_indexes)
                                
                        def __len__(self):
                            return int(len(self.videos_path) / self.batch_size)
                            
                        def __getitem__(self, index):
                            # Generate indexes of the batch
                            indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]
                            
                            # Find list of IDs
                            batch_ids = [self.videos_path[k] for k in indexes]
                            
                            # Generate data
                            X, y = self.__data_generation(batch_ids)
                            
                            if self.to_fit:
                                return X, y
                            else:
                                return X
                        
                        def __data_generation(self, batch_ids):
                            data = np.empty(self.data_shape)
                            classes = np.empty(self.data_shape[0], dtype=int)
                            
                            for i, instance_id in enumerate(batch_ids):
                                x = np.load(instance_id[0])['frames']
                                data[i,] = x
                                classes[i] = class_mapping[instance_id[1]]
                            
                            self.classes += list(classes)
                            
                            return (
                                data,
                                to_categorical(
                                    classes,
                                    num_classes=self.n_classes
                                )
                            )
                    """
                ).format(generator_name=generator_name)

        elif self.cropping_strategy == 'random':
            if self.video_training:
                return dedent(
                    """     
                    class {generator_name}(object):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            
                        def next_video(self):
                            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                            # Initialization
                            data = np.empty(self.data_shape)
                            classes = np.empty((self.batch_size), dtype=int)
                            
                            # Generate data    
                            while True:
                                i = 0
                                if self.shuffle:
                                    random.shuffle(self.videos_path)
                                
                                for _file, cls in self.videos_path:
                                    x = np.load(_file)['frames']
                                    h_init = random.randint({height})
                                    w_init = random.randint({width})
                                    data[i,] = x[{frames}, h_init:h_init+{dim1}, w_init:w_init+{dim2}, {channel}] 
                
                                    classes[i] = class_mapping[cls]
                                    i += 1
                                    
                                    if i % self.batch_size == 0:
                                        yield (
                                            data,
                                            to_categorical(
                                                classes,
                                                num_classes=self.n_classes
                                            )
                                        )
                                        self.classes += list(classes)
                                        i = 0
                                if i > 0:
                                    yield (
                                        data[:i],
                                        to_categorical(
                                            classes[:i],
                                            num_classes=self.n_classes
                                        )
                                    )
                                    self.classes += list(classes[:i])
                    
                    {class_mapping}
                
                    def video_path_reader(path='', validation_split=0.0):
                        if path:
                            classes = [x[0] for x in walk(path)][1:]
                        else:
                            raise ValueError(gettext('Data set path is invalid.'))
                        
                        if classes:
                            files = []
                            id = 0
                            for class_path in classes:
                                cls = class_path.split('/')[-1] # Get only class name
                                if not cls in class_mapping:
                                    class_mapping[cls] = id
                                    id += 1
                    
                                files += [(class_path+'/'+f, cls) for f in listdir(class_path) if isfile(join(class_path, f))]
                                
                            if validation_split:
                                _index = int(len(files)*validation_split)
                                
                                training_files = files[_index:]
                                validation_files = files[0:_index]
                                
                                return {_return}
                            else:
                                return files
                    """
                ).format(generator_name=generator_name,
                         height=self.random_height,
                         width=self.random_width,
                         frames=self.random_frames,
                         channel=self.random_channel,
                         dim1=self.dimensions[1],
                         dim2=self.dimensions[2],
                         _return='{"training": training_files, '
                                 '"validation": validation_files}',
                         class_mapping='class_mapping = {}')

            elif self.video_validation:
                return dedent(
                    """   
                    class {generator_name}(object):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            
                        def next_video(self):
                            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                            # Initialization
                            data = np.empty(self.data_shape)
                            classes = np.empty((self.batch_size), dtype=int)
                            
                            # Generate data  
                            while True:
                                i = 0
                                if self.shuffle:
                                    random.shuffle(self.videos_path)
                                    
                                for _file, cls in self.videos_path:
                                    x = np.load(_file)['frames']
                                    h_init = random.randint({height})
                                    w_init = random.randint({width})
                                    data[i,] = x[{frames}, h_init:h_init+{dim1}, w_init:w_init+{dim2}, {channel}] 
                
                                    classes[i] = class_mapping[cls]
                                    i += 1
                                    
                                    if i % self.batch_size == 0:
                                        yield (
                                            data,
                                            to_categorical(
                                                classes,
                                                num_classes=self.n_classes
                                            )
                                        )
                                        self.classes += list(classes)
                                        i = 0
                                if i > 0:
                                    yield (
                                        data[:i],
                                        to_categorical(
                                            classes[:i],
                                            num_classes=self.n_classes
                                        )
                                    )
                                    self.classes += list(classes[:i])
                    """
                ).format(generator_name=generator_name,
                         height=self.random_height,
                         width=self.random_width,
                         frames=self.random_frames,
                         channel=self.random_channel,
                         dim1=self.dimensions[1],
                         dim2=self.dimensions[2])
            elif self.video_test:
                return dedent(
                    """   
                    class {generator_name}(object):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            
                        def next_video(self):
                            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                            # Initialization
                            data = np.empty(self.data_shape)
                            classes = np.empty((self.batch_size), dtype=int)
                            
                            # Generate data  
                            while True:
                                i = 0
                                if self.shuffle:
                                    random.shuffle(self.videos_path)
                                    
                                for _file, cls in self.videos_path:
                                    x = np.load(_file)['frames']
                                    h_init = random.randint({height})
                                    w_init = random.randint({width})
                                    data[i,] = x[{frames}, h_init:h_init+{dim1}, w_init:w_init+{dim2}, {channel}] 
                
                                    classes[i] = class_mapping[cls]
                                    i += 1
                                    
                                    if i % self.batch_size == 0:
                                        if len(self.classes) < len(self.videos_path):
                                            self.classes += list(classes)
                                            
                                        yield (
                                            data,
                                            to_categorical(
                                                classes,
                                                num_classes=self.n_classes
                                            )
                                        )
                                        i = 0
                                if i > 0:
                                    if len(self.classes) < len(self.videos_path):
                                        self.classes += list(classes[:i])
                                        
                                    yield (
                                        data[:i],
                                        to_categorical(
                                            classes[:i],
                                            num_classes=self.n_classes
                                        )
                                    )
                    """
                ).format(generator_name=generator_name,
                         height=self.random_height,
                         width=self.random_width,
                         frames=self.random_frames,
                         channel=self.random_channel,
                         dim1=self.dimensions[1],
                         dim2=self.dimensions[2])

        elif self.cropping_strategy == 'center':
            if self.video_training:
                return dedent(
                    """     
                    class {generator_name}(object):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            
                        def next_video(self):
                            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                            # Initialization
                            data = np.empty(self.data_shape)
                            classes = np.empty((self.batch_size), dtype=int)
                            
                            # Generate data    
                            while True:
                                i = 0
                                if self.shuffle:
                                    random.shuffle(self.videos_path)
                                    
                                for _file, cls in self.videos_path:
                                    x = np.load(_file)['frames']
                                    data[i,] = x[{frames}, {height}, {width}, {channel}] 
                
                                    classes[i] = class_mapping[cls]
                                    i += 1
                                    
                                    if i % self.batch_size == 0:
                                        yield (
                                            data,
                                            to_categorical(
                                                classes,
                                                num_classes=self.n_classes
                                            )
                                        )
                                        self.classes += list(classes)
                                        i = 0
                                if i > 0:
                                    yield (
                                        data[:i],
                                        to_categorical(
                                            classes[:i],
                                            num_classes=self.n_classes
                                        )
                                    )
                                    self.classes += list(classes[:i])
                    
                    {class_mapping}
                
                    def video_path_reader(path='', validation_split=0.0):
                        if path:
                            classes = [x[0] for x in walk(path)][1:]
                        else:
                            raise ValueError(gettext('Data set path is invalid.'))
                        
                        if classes:
                            files = []
                            id = 0
                            for class_path in classes:
                                cls = class_path.split('/')[-1] # Get only class name
                                if not cls in class_mapping:
                                    class_mapping[cls] = id
                                    id += 1
                    
                                files += [(class_path+'/'+f, cls) for f in listdir(class_path) if isfile(join(class_path, f))]
                                
                            if validation_split:
                                _index = int(len(files)*validation_split)
                                
                                training_files = files[_index:]
                                validation_files = files[0:_index]
                                
                                return {_return}
                            else:
                                return files
                    """
                ).format(generator_name=generator_name,
                         height=self.height,
                         width=self.width,
                         frames=self.frames,
                         channel=self.channel,
                         class_mapping='class_mapping = {}',
                         _return='{"training": training_files, '
                                 '"validation": validation_files}')
            elif self.video_validation:
                return dedent(
                    """     
                    class {generator_name}(object):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            
                        def next_video(self):
                            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                            # Initialization
                            data = np.empty(self.data_shape)
                            classes = np.empty((self.batch_size), dtype=int)
                            
                            # Generate data
                            while True:
                                i = 0
                                if self.shuffle:
                                    random.shuffle(self.videos_path)
                                    
                                for _file, cls in self.videos_path:
                                    x = np.load(_file)['frames']
                                    data[i,] = x[{frames}, {height}, {width}, {channel}] 
                
                                    classes[i] = class_mapping[cls]
                                    i += 1
                                    
                                    if i % self.batch_size == 0:
                                        yield (
                                            data,
                                            to_categorical(
                                                classes,
                                                num_classes=self.n_classes
                                            )
                                        )
                                        self.classes += list(classes)
                                        i = 0
                                if i > 0:
                                    yield (
                                        data[:i],
                                        to_categorical(
                                            classes[:i],
                                            num_classes=self.n_classes
                                        )
                                    )
                                    self.classes += list(classes[:i])
                    """
                ).format(generator_name=generator_name,
                         height=self.height,
                         width=self.width,
                         frames=self.frames,
                         channel=self.channel,
                         class_mapping='class_mapping = {}')

            elif self.video_test:
                return dedent(
                    """     
                    class {generator_name}(object):
                        def __init__(self, videos_path=[],
                                           batch_size=16,
                                           data_shape=None,
                                           n_classes=0, 
                                           shuffle=True):
                                           
                            self.videos_path = videos_path
                            self.batch_size = batch_size
                            self.data_shape = data_shape
                            self.n_classes = n_classes
                            self.shuffle = shuffle
                            self.classes = []
                            
                        def next_video(self):
                            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                            # Initialization
                            data = np.empty(self.data_shape)
                            classes = np.empty((self.batch_size), dtype=int)
                            
                            # Generate data   
                            while True:
                                i = 0
                                if self.shuffle:
                                    random.shuffle(self.videos_path)
                                    
                                for _file, cls in self.videos_path:
                                    x = np.load(_file)['frames']
                                    data[i,] = x[{frames}, {height}, {width}, {channel}] 
                
                                    classes[i] = class_mapping[cls]
                                    i += 1
                                    
                                    if i % self.batch_size == 0:
                                        if len(self.classes) < len(self.videos_path):
                                            self.classes += list(classes)
                                            
                                        yield (
                                            data,
                                            to_categorical(
                                                classes,
                                                num_classes=self.n_classes
                                            )
                                        )
                                        i = 0
                                if i > 0:
                                    if len(self.classes) < len(self.videos_path):
                                        self.classes += list(classes[:i])
                                        
                                    yield (
                                        data[:i],
                                        to_categorical(
                                            classes[:i],
                                            num_classes=self.n_classes
                                        )
                                    )
                    """
                ).format(generator_name=generator_name,
                         height=self.height,
                         width=self.width,
                         frames=self.frames,
                         channel=self.channel,
                         class_mapping='class_mapping = {}')

    def generate_code(self):
        data_shape = [self.batch_size]
        for dim in self.dimensions:
            data_shape.append(dim)
        data_shape.append(self.channels)
        data_shape = tuple(data_shape)

        if self.video_training:
            if self.validation_split == 0:
                return dedent(
                    """
                    training_videos_path = video_path_reader(
                        path={path}
                    )
                    
                    training_video_generator = VideoGeneratorTraining(
                        videos_path=training_videos_path,
                        batch_size={batch_size},
                        data_shape={data_shape},
                        n_classes=len(class_mapping),
                        shuffle={shuffle}
                    )
                    
                    train_{var_name} = training_video_generator#.next_video()
                    
                    predict_training_video_generator = VideoGeneratorTraining(
                        videos_path=training_video_generator.videos_path,
                        batch_size={batch_size},
                        data_shape={data_shape},
                        n_classes=len(class_mapping),
                        shuffle=False
                    )

                    predict_train_{var_name} = predict_training_video_generator#.next_video()
                    """
                ).format(var_name=self.var_name,
                         path=self.video_training,
                         subset='training',
                         batch_size=self.batch_size,
                         data_shape=data_shape,
                         shuffle=self.shuffle)
            else:
                return dedent(
                    """
                    videos_path = video_path_reader(
                        path={path},
                        validation_split={validation_split}
                    )
                    
                    training_video_generator = VideoGenerator(
                        videos_path=videos_path[{subset_training}],
                        batch_size={batch_size},
                        data_shape={data_shape},
                        n_classes=len(class_mapping),
                        shuffle={shuffle}
                    )
                    
                    train_{var_name} = training_video_generator#.next_video()
                    
                    predict_training_video_generator = VideoGeneratorTraining(
                        videos_path=training_video_generator.videos_path[{subset_training}],
                        batch_size={batch_size},
                        data_shape={data_shape},
                        n_classes=len(class_mapping),
                        shuffle=False
                    )

                    predict_train_{var_name} = predict_training_video_generator#.next_video()
                    
                    validation_video_generator = VideoGenerator(
                        videos_path=videos_path[{subset_validation}],
                        batch_size={batch_size},
                        data_shape={data_shape},
                        n_classes=len(class_mapping),
                        shuffle={shuffle}
                    )
                    
                    validation_{var_name} = validation_video_generator#.next_video()
                    
                    predict_validation_video_generator = VideoGenerator(
                        videos_path=validation_video_generator.videos_path[{subset_validation}],
                        batch_size={batch_size},
                        data_shape={data_shape},
                        n_classes=len(class_mapping),
                        shuffle=False
                    )

                    predict_validation_{var_name} = predict_validation_video_generator#.next_video()
                    """
                ).format(var_name=self.var_name,
                         path=self.video_training,
                         validation_split=self.validation_split,
                         subset_training='training',
                         subset_validation='validation',
                         batch_size=self.batch_size,
                         data_shape=data_shape,
                         shuffle=self.shuffle)

        if self.video_validation:
            return dedent(
                """
                validation_videos_path = video_path_reader(
                        path={path}
                )
                    
                validation_video_generator = VideoGeneratorValidation(
                    videos_path=validation_videos_path,
                    batch_size={batch_size},
                    data_shape={data_shape},
                    n_classes=len(class_mapping),
                    shuffle={shuffle}
                )
                
                validation_{var_name} = validation_video_generator#.next_video()
                
                predict_validation_video_generator = VideoGeneratorValidation(
                    videos_path=validation_video_generator.videos_path,
                    batch_size={batch_size},
                    data_shape={data_shape},
                    n_classes=len(class_mapping),
                    shuffle=False
                )

                predict_validation_{var_name} = predict_validation_video_generator#.next_video()
                """
            ).format(var_name=self.var_name,
                     path=self.video_validation,
                     batch_size=self.batch_size,
                     data_shape=data_shape,
                     shuffle=self.shuffle)

        if self.video_test:
            return dedent(
                """
                test_videos_path = video_path_reader(
                        path={path}
                )
                    
                test_video_generator = VideoGeneratorTest(
                    videos_path=test_videos_path,
                    batch_size={batch_size},
                    data_shape={data_shape},
                    n_classes=len(class_mapping),
                    shuffle={shuffle}
                )
                
                test_{var_name} = test_video_generator#.next_video()
                
                """
            ).format(var_name=self.var_name,
                     path=self.video_test,
                     batch_size=self.batch_size,
                     data_shape=data_shape,
                     shuffle=self.shuffle)
