# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


class FileReader(Operation):
    FILE_OR_FOLDER_PARAM = 'file_or_folder'
    OUT_CODE_PARAM = 'out_code'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.file_or_folder = parameters.get(self.FILE_OR_FOLDER_PARAM, None)
        self.out_code = True if \
            int(parameters.get(self.OUT_CODE_PARAM, 0)) == 1 else False

        self.has_code = not self.out_code
        self.has_external_python_code_operation = self.out_code

        self.var_name = ""
        self.task_name = self.parameters.get('task').get('name')

        supported_formats = ('NPY', 'SAV', 'DATA_FOLDER')

        if self.file_or_folder != 0 and self.file_or_folder is not None:
            self.metadata = self.get_data_source(
                data_source_id=self.file_or_folder)

            if self.metadata.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported file format: {}').format(
                    self.metadata.get('format')))

            self.format = self.metadata.get('format')

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def get_data_source(self, data_source_id):
        # Retrieve metadata from Limonero.
        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']

        metadata = limonero_service.get_data_source_info(
            limonero_config['url'], str(limonero_config['auth_token']),
            str(data_source_id))

        if not metadata.get('url'):
            raise ValueError(
                gettext('Incorrect data source configuration (empty url)'))

        return metadata

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.file_or_folder is not None:
            self.file_or_folder = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata.get('storage').get('url'),
                file_url=self.metadata.get('url')
            )
            self.file_or_folder = self.file_or_folder.replace('file://', '')

    def generate_code(self):
        options = []
        if self.file_or_folder:
            options.append("""                
                {var_name} = {file_or_folder}
                """.format(var_name=self.var_name,
                           file_or_folder=self.file_or_folder))

        return dedent('\n'.join(options))


class ImageReader(Operation):
    TRAIN_IMAGES_PARAM = 'train_images'
    VALIDATION_IMAGES_PARAM = 'validation_images'
    TEST_IMAGES_PARAM = 'test_images'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.train_images = parameters.get(self.TRAIN_IMAGES_PARAM, None)
        self.validation_images = parameters.get(self.VALIDATION_IMAGES_PARAM,
                                                None)
        self.test_images = parameters.get(self.TEST_IMAGES_PARAM, None)

        self.has_code = True
        self.var_name = ""
        self.task_name = self.parameters.get('task').get('name')

        supported_formats = ('IMAGE_FOLDER',)

        if self.train_images != 0 and self.train_images is not None:
            self.metadata_train = self.get_data_source(
                data_source_id=self.train_images)

            if self.metadata_train.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported image format: {}').format(
                    self.metadata_train.get('format')))

            self.format = self.metadata_train.get('format')

        if self.validation_images != 0 and self.validation_images is not None:
            self.metadata_validation = self.get_data_source(
                data_source_id=self.validation_images)

            if self.metadata_validation.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported image format: {}').format(
                    self.metadata_validation.get('format')))

            self.format = self.metadata_validation.get('format')
        else:
            self.metadata_validation = None

        if self.test_images != 0 and self.test_images is not None:
            self.metadata_test = self.get_data_source(
                data_source_id=self.test_images)

            if self.metadata_test.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported image format: {}').format(
                    self.metadata_test.get('format')))

            self.format = self.metadata_test.get('format')
        else:
            self.metadata_test = None

        if self.metadata_validation is not None:
            if not (self.metadata_train.get('format') ==
                    self.metadata_validation.get('format')):
                raise ValueError(gettext('Training and validation images files '
                                         'are in different formats.'))

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def get_data_source(self, data_source_id):
        # Retrieve metadata from Limonero.
        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']

        metadata = limonero_service.get_data_source_info(
            limonero_config['url'], str(limonero_config['auth_token']),
            str(data_source_id))

        if not metadata.get('url'):
            raise ValueError(
                gettext('Incorrect data source configuration (empty url)'))

        return metadata

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.train_images is not None:
            self.train_images = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_train.get('storage').get('url'),
                file_url=self.metadata_train.get('url')
            )
            self.train_images = self.train_images.replace('file://', '')

        if self.validation_images is not None:
            self.validation_images = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_validation.get('storage').get('url'),
                file_url=self.metadata_validation.get('url')
            )
            self.validation_images = self.validation_images.replace(
                'file://', '')

        if self.test_images is not None:
            self.test_images = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_test.get('storage').get('url'),
                file_url=self.metadata_test.get('url')
            )
            self.test_images = self.test_images.replace('file://', '')

    def generate_code(self):
        options = []
        
        if self.train_images:
            options.append("""                
                {var_name}_train_image = {train_images}
                """.format(var_name=self.var_name,
                           train_images=self.train_images))

        if self.validation_images:
            options.append("""                
                {var_name}_validation_image = {validation_images}
                """.format(var_name=self.var_name,
                           validation_images=self.validation_images))

        if self.test_images:
            options.append("{var_name}_test_image = {test_images}".format(
                var_name=self.var_name, test_images=self.test_images))

        return dedent('\n'.join(options))


class SequenceReader(Operation):
    TRAIN_SEQUENCE_PARAM = 'train_sequence'
    VALIDATION_SEQUENCE_PARAM = 'validation_sequence'
    TEST_SEQUENCE_PARAM = 'test_sequence'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.train_sequence = parameters.get(self.TRAIN_SEQUENCE_PARAM, None)
        self.validation_sequence = parameters.get(self.VALIDATION_SEQUENCE_PARAM,
                                                  None)
        self.test_sequence = parameters.get(self.TEST_SEQUENCE_PARAM, None)

        self.has_code = True
        self.var_name = ""
        self.task_name = self.parameters.get('task').get('name')

        supported_formats = ('IMAGE_FOLDER', 'VIDEO_FOLDER', 'SEQUENCE_FOLDER')

        if self.train_sequence != 0 and self.train_sequence is not None:
            self.metadata_train = self.get_data_source(
                data_source_id=self.train_sequence)

            if self.metadata_train.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported format: {}').format(
                    self.metadata_train.get('format')))

            self.format = self.metadata_train.get('format')

        if self.validation_sequence != 0 and self.validation_sequence is not None:
            self.metadata_validation = self.get_data_source(
                data_source_id=self.validation_sequence)

            if self.metadata_validation.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported format: {}').format(
                    self.metadata_validation.get('format')))

            self.format = self.metadata_validation.get('format')
        else:
            self.metadata_validation = None

        if self.test_sequence != 0 and self.test_sequence is not None:
            self.metadata_test = self.get_data_source(
                data_source_id=self.test_sequence)

            if self.metadata_test.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported format: {}').format(
                    self.metadata_test.get('format')))

            self.format = self.metadata_test.get('format')
        else:
            self.metadata_test = None

        if self.metadata_validation is not None:
            if not (self.metadata_train.get('format') ==
                    self.metadata_validation.get('format')):
                raise ValueError(gettext('Training and validation files '
                                         'are in different formats.'))

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def get_data_source(self, data_source_id):
        # Retrieve metadata from Limonero.
        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']

        metadata = limonero_service.get_data_source_info(
            limonero_config['url'], str(limonero_config['auth_token']),
            str(data_source_id))

        if not metadata.get('url'):
            raise ValueError(
                gettext('Incorrect data source configuration (empty url)'))

        return metadata

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.train_sequence is not None:
            self.train_sequence = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_train.get('storage').get('url'),
                file_url=self.metadata_train.get('url')
            )
            self.train_sequence = self.train_sequence.replace('file://', '')

        if self.validation_sequence is not None:
            self.validation_sequence = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_validation.get('storage').get('url'),
                file_url=self.metadata_validation.get('url')
            )
            self.validation_sequence = self.validation_sequence.replace(
                'file://', '')

        if self.test_sequence is not None:
            self.test_sequence = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_test.get('storage').get('url'),
                file_url=self.metadata_test.get('url')
            )
            self.test_sequence = self.test_sequence.replace('file://', '')

    def generate_code(self):
        options = []

        if self.train_sequence:
            options.append("""                
                {var_name}_train_sequence = {train_sequences}
                """.format(var_name=self.var_name,
                           train_sequences=self.train_sequence))

        if self.validation_sequence:
            options.append("""                
                {var_name}_validation_sequence = {validation_sequences}
                """.format(var_name=self.var_name,
                           validation_sequences=self.validation_sequence))

        if self.test_sequence:
            options.append("{var_name}_test_sequence = {test_sequences}".format(
                var_name=self.var_name, test_sequences=self.test_sequence))

        return dedent('\n'.join(options))


class VideoReader(Operation):
    TRAIN_VIDEOS_PARAM = 'training_videos'
    VALIDATION_VIDEOS_PARAM = 'validation_videos'
    TEST_VIDEOS_PARAM = 'test_videos'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.TRAIN_VIDEOS_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.TRAIN_VIDEOS_PARAM))

        self.train_videos = parameters.get(self.TRAIN_VIDEOS_PARAM, None)
        self.validation_videos = parameters.get(self.VALIDATION_VIDEOS_PARAM,
                                                None)

        self.test_videos = parameters.get(self.TEST_VIDEOS_PARAM, None)

        self.has_code = True
        self.var_name = ""
        self.task_name = self.parameters.get('task').get('name')

        supported_formats = ('VIDEO_FOLDER',)

        if self.train_videos != 0 and self.train_videos is not None:
            self.metadata_train = self.get_data_source(
                data_source_id=self.train_videos)

            if self.metadata_train.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported format: {}').format(
                    self.metadata_train.get('format')))

            self.format = self.metadata_train.get('format')

        if self.validation_videos != 0 and self.validation_videos is not None:
            self.metadata_validation = self.get_data_source(
                data_source_id=self.validation_videos)

            if self.metadata_validation.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported format: {}').format(
                    self.metadata_validation.get('format')))

            self.format = self.metadata_validation.get('format')
        else:
            self.metadata_validation = None

        if self.test_videos != 0 and self.test_videos is not None:
            self.metadata_test = self.get_data_source(
                data_source_id=self.test_videos)

            if self.metadata_test.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported format: {}').format(
                    self.metadata_test.get('format')))

            self.format = self.metadata_test.get('format')
        else:
            self.metadata_test = None

        if self.metadata_validation is not None:
            if not (self.metadata_train.get('format') ==
                    self.metadata_validation.get('format')):
                raise ValueError(gettext('Training and validation files '
                                         'are in different formats.'))

        if self.metadata_test is not None:
            if not (self.metadata_train.get('format') ==
                    self.metadata_test.get('format')):
                raise ValueError(gettext('Training and test files '
                                         'are in different formats.'))

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def get_data_source(self, data_source_id):
        # Retrieve metadata from Limonero.
        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']

        metadata = limonero_service.get_data_source_info(
            limonero_config['url'], str(limonero_config['auth_token']),
            str(data_source_id))

        if not metadata.get('url'):
            raise ValueError(
                gettext('Incorrect data source configuration (empty url)'))

        return metadata

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.train_videos is not None:
            self.train_videos = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_train.get('storage').get('url'),
                file_url=self.metadata_train.get('url')
            )
            self.train_videos = self.train_videos.replace('file://', '')

        if self.validation_videos is not None:
            self.validation_videos = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_validation.get('storage').get('url'),
                file_url=self.metadata_validation.get('url')
            )
            self.validation_videos = self.validation_videos.replace('file://', '')

        if self.test_videos is not None:
            self.test_videos = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_test.get('storage').get('url'),
                file_url=self.metadata_test.get('url')
            )
            self.test_videos = self.test_videos.replace('file://', '')

    def generate_code(self):
        if self.train_videos and self.validation_videos and self.test_videos:
            return dedent(
                """                
                {var_name}_train_video = {train_videos}
                {var_name}_validation_video = {validation_videos}
                {var_name}_test_video = {test_videos}
                """.format(var_name=self.var_name,
                           train_videos=self.train_videos,
                           validation_videos=self.validation_videos,
                           test_videos=self.test_videos)
            )
        elif self.train_videos and self.validation_videos:
            return dedent(
                """                
                {var_name}_train_video = {train_videos}
                {var_name}_validation_video = {validation_videos}
                """.format(var_name=self.var_name,
                           train_videos=self.train_videos,
                           validation_videos=self.validation_videos)
            )
        elif self.train_videos and self.test_videos:
            return dedent(
                """                
                {var_name}_train_video = {train_videos}
                {var_name}_test_video = {test_videos}
                """.format(var_name=self.var_name,
                           train_videos=self.train_videos,
                           test_videos=self.test_videos)
            )
        elif self.train_videos:
            return dedent(
                """
                {var_name}_train_video = {train_videos}
                """.format(var_name=self.var_name,
                           train_videos=self.train_videos)
            )
