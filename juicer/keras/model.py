# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


class EvaluateModel(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.has_code = True
        self.var_name = ""
        self.task_name = self.parameters.get('task').get('name')
        self.model = ''
        self.generator = 'test_video_generator'
        self.iterator = ''
        self.task_id = self.parameters.get('task').get('id')
        self.is_video_generator = False

        self.treatment()

        self.import_code = {'layer': [],
                            'callbacks': [],
                            'model': [],
                            'preprocessing_image': [],
                            'others': []}

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        parents_by_port = self.parameters.get('parents_by_port', [])
        if len(parents_by_port) == 2:
            if parents_by_port[0][0] == 'model':
                self.model = parents_by_port[0][1]
                self.iterator = parents_by_port[1][1]
            else:
                self.model = parents_by_port[1][1]
                self.iterator = parents_by_port[0][1]
        else:
            raise ValueError(gettext('You need to correctly specify the '
                                     'testing data and the model.'))

        self.model = convert_variable_name(self.model)
        self.iterator = convert_variable_name(self.iterator)

        if 'video-generator' in self.parameters.get('parents_slug'):
            self.is_video_generator = True

    def generate_code(self):
        if self.is_video_generator:
            return dedent(
                """
                    batch_size = {generator}.batch_size
                    number_of_videos = len({generator}.videos_path)
                    if number_of_videos % batch_size == 0:
                        steps = number_of_videos // batch_size
                    else:
                        steps = (number_of_videos // batch_size) + 1
                        
                    model_evaluate = {model}.evaluate_generator(
                        generator=test_{iterator},
                        steps=steps
                    )
                    loss_acc_values = 'Loss: ' + ("%.3f" % model_evaluate[0]) 
                    loss_acc_values += '  --  Accuracy: ' + ("%.3f" % model_evaluate[1]) 
                    
                    message = '\\n<h5>Classification Report - Testing</h5>'
                    message += '<pre>' + str(loss_acc_values) + '</pre>'
                    emit_event(name='update task',
                        message=message,
                        type='HTML',
                        status='RESULTS',
                        identifier='{task_id}'
                    ) 
                    
                    model_predict = {model}.predict_generator(
                        generator=test_{iterator},
                        steps=steps
                    )
                    
                    predictions_to_matrix = np.argmax(model_predict, axis=1)
                    
                    report = classification_report(
                        y_true={generator}.classes,
                        y_pred=predictions_to_matrix,
                        labels=list(class_mapping.values()),
                        target_names=list(class_mapping.keys()),
                        digits=3,
                        output_dict=False
                    )
                    
                    message = '<pre>' + report + '</pre>'
                    emit_event(name='update task',
                        message=message,
                        type='HTML',
                        status='RESULTS',
                        identifier='{task_id}'
                    ) 
                """.format(var_name=self.var_name,
                           model=self.model,
                           generator=self.generator,
                           iterator=self.iterator,
                           task_id=self.task_id)
            )
        else:
            return dedent(
                """
                    batch_size = {generator}.batch_size
                    number_of_videos = len({generator}.videos_path)
                    if number_of_videos % batch_size == 0:
                        steps = number_of_videos // batch_size
                    else:
                        steps = (number_of_videos // batch_size) + 1
                        
                    model_evaluate = {model}.evaluate_generator(
                        generator=test_{iterator},
                        steps=steps
                    )
                    loss_acc_values = 'Loss: ' + ("%.3f" % model_evaluate[0]) 
                    loss_acc_values += '  --  Accuracy: ' + ("%.3f" % model_evaluate[1]) 
                    
                    message = '\\n<h5>Classification Report - Testing</h5>'
                    message += '<pre>' + loss_acc_values + '</pre>'
                    emit_event(name='update task',
                        message=message,
                        type='HTML',
                        status='RESULTS',
                        identifier='{task_id}'
                    ) 
                    
                    model_predict = {model}.predict_generator(
                        generator=test_{iterator},
                        steps=steps
                    )
                    
                    predictions_to_matrix = np.argmax(model_predict, axis=1)
                    
                    target_names = list({generator}.class_indices.keys())
                    labels = list({generator}.class_indices.values())
                    report = classification_report(
                        y_true={generator}.classes,
                        y_pred=predictions_to_matrix,
                        labels=labels,
                        target_names=target_names,
                        output_dict=False
                    )
                    
                    message = '<pre>' + report + '</pre>'
                    emit_event(name='update task',
                        message=message,
                        type='HTML',
                        status='RESULTS',
                        identifier='{task_id}'
                    ) 
                """.format(var_name=self.var_name,
                           model=self.model,
                           generator=self.generator,
                           iterator=self.iterator,
                           task_id=self.task_id)
            )


class FitGenerator(Operation):
    # Fit Generator
    STEPS_PER_EPOCH_PARAM = 'steps_per_epoch'
    EPOCHS_PARAM = 'epochs'
    VERBOSE_PARAM = 'verbose'
    CALLBACKS_PARAM = 'callbacks'
    VALIDATION_DATA_PARAM = 'validation_data'
    VALIDATION_STEPS_PARAM = 'validation_steps'
    VALIDATION_FREQ_PARAM = 'validation_freq'
    CLASS_WEIGHT_PARAM = 'class_weight'
    MAX_QUEUE_SIZE_PARAM = 'max_queue_size'
    WORKERS_PARAM = 'workers'
    USE_MULTIPROCESSING_PARAM = 'use_multiprocessing'
    SHUFFLE_PARAM = 'shuffle'
    INITIAL_EPOCH_PARAM = 'initial_epoch'

    #Save Model
    SAVE_ENABLED_PARAM = 'save_enabled'
    SAVE_STORAGE_PARAM = 'storage'
    SAVE_NAME_PARAM = 'save_name'
    SAVE_ACTION_IF_EXISTS_PARAM = 'action_if_exists'
    SAVE_WEIGHTS_ONLY_PARAM = 'save_weights_only'
    SAVE_METRICS_PARAM = 'save_metrics'
    SAVE_SUBSET_PARAM = 'save_subset'

    # Classification report
    CLASSIFICATION_REPORT_PARAM = 'classification_report'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        # Fit Generator
        self.steps_per_epoch = parameters.get(self.STEPS_PER_EPOCH_PARAM,
                                              None)
        self.epochs = parameters.get(self.EPOCHS_PARAM, None)
        self.validation_data = parameters.get(self.VALIDATION_DATA_PARAM,
                                              None)
        self.validation_steps = parameters.get(self.VALIDATION_STEPS_PARAM,
                                               None)
        self.validation_freq = parameters.get(self.VALIDATION_FREQ_PARAM,
                                              None)
        self.class_weight = parameters.get(self.CLASS_WEIGHT_PARAM,
                                           None)
        self.max_queue_size = parameters.get(self.MAX_QUEUE_SIZE_PARAM,
                                             None)
        self.workers = int(parameters.get(self.WORKERS_PARAM, None))
        self.use_multiprocessing = parameters.get(
            self.USE_MULTIPROCESSING_PARAM, None)
        self.shuffle = parameters.get(self.SHUFFLE_PARAM, None)
        self.initial_epoch = parameters.get(self.INITIAL_EPOCH_PARAM,
                                            None)

        self.callbacks = [
            {'key': 'TensorBoard', 'value': 'TensorBoard'},
            {'value': 'History', 'key': 'History'}
        ]

        # Save params
        self.save_enabled = parameters.get(self.SAVE_ENABLED_PARAM, None)
        self.save_storage = parameters.get(self.SAVE_STORAGE_PARAM, None)
        self.save_name = parameters.get(self.SAVE_NAME_PARAM, None)
        self.save_action_if_exists = parameters.get(
            self.SAVE_ACTION_IF_EXISTS_PARAM, None)
        self.save_weights_only = parameters.get(self.SAVE_WEIGHTS_ONLY_PARAM, None)
        self.save_metrics = parameters.get(self.SAVE_METRICS_PARAM, None)
        self.save_subset = parameters.get(self.SAVE_SUBSET_PARAM, None)

        self.classification_report = parameters.get(
            self.CLASSIFICATION_REPORT_PARAM, None)

        self.model = None

        self.callback_code = ''

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True
        self.show_results = True

        self.add_functions_required_compile = ""
        self.add_functions_required_fit_generator = ""

        self.output_task_id = self.parameters.get('task').get('id')

        if self.STEPS_PER_EPOCH_PARAM not in parameters or self.steps_per_epoch is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.STEPS_PER_EPOCH_PARAM))

        if self.EPOCHS_PARAM not in parameters or self.epochs is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.EPOCHS_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])

        if len(self.parents_by_port) < 2:
            raise ValueError(gettext('The operation needs training and model.'))

        self.input_layers = []
        self.output_layers = []
        self.train_generator = None
        self.validation_generator = None

        self.is_video_generator = False

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': 'Model',
                            'preprocessing_image': None,
                            'others': None}

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

    def treatment(self):
        #self.var_name = convert_variable_name(self.task_name)
        self.task_name = convert_variable_name(self.task_name)

        validation = ''
        for parent in self.parents_by_port:
            if str(parent[0]) == 'train-generator':
                self.train_generator = 'train_{var_name}' \
                    .format(var_name=convert_variable_name(parent[1]))
                if validation == '':
                    validation = convert_variable_name(parent[1])
            elif str(parent[0]) == 'validation-generator':
                validation = convert_variable_name(parent[1])
            elif str(parent[0]) == 'model':
                self.model = convert_variable_name(parent[1])
                self.var_name = self.model

        if validation:
            self.validation_generator = 'validation_{var_name}' \
                .format(var_name=validation)

        if self.train_generator is None:
            self.show_results = False
            if self.validation_generator:
                raise ValueError(gettext('It is not possible to use only '
                                         'validation data.'))

        if self.model is None:
            if self.validation_generator:
                raise ValueError(gettext('It is necessary to inform the model.')
                                 )

        self.shuffle = True if int(self.shuffle) == 1 else False
        self.use_multiprocessing = True if int(
            self.use_multiprocessing) == 1 else False

        if 'video-generator' in self.parameters.get('parents_slug'):
            self.is_video_generator = True
            self.shuffle = False# Used not to impact the shuffle value in the video generator.
            if self.workers > 1 or self.use_multiprocessing:
                import warnings
                warnings.warn('Parameters changed: use_multiprocessing=False, '
                              'workers=1 -- The video generator does not (yet)'
                              'support multiprocessing.')
                self.workers = 1
                self.use_multiprocessing = False

        # Fit Generator
        functions_required_fit_generator = []
        if self.train_generator is not None:
            self.train_generator = """    generator={train_generator}""" \
                .format(train_generator=self.train_generator)
            functions_required_fit_generator.append(self.train_generator)

        if self.steps_per_epoch is not None:
            functions_required_fit_generator.append(
                """steps_per_epoch={steps_per_epoch}""".format(
                    steps_per_epoch=self.steps_per_epoch))
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.STEPS_PER_EPOCH_PARAM))

        epochs = self.epochs
        if self.epochs is not None:
            self.epochs = """epochs={epochs}""".format(epochs=self.epochs)
            functions_required_fit_generator.append(self.epochs)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.EPOCHS_PARAM))

        # SAVE MODEL (USING CALLBACK ModelCheckpoint)
        self.save_enabled = True if int(self.save_enabled) == 1 else False
        self.save_weights_only = True if int(self.save_weights_only) == 1 else False

        if self.save_enabled:
            if self.SAVE_STORAGE_PARAM not in self.parameters:
                raise ValueError(gettext('Parameter {} is required')
                                 .format(self.SAVE_STORAGE_PARAM))
            if self.SAVE_NAME_PARAM not in self.parameters:
                raise ValueError(gettext('Parameter {} is required')
                                 .format(self.SAVE_NAME_PARAM))
            if self.SAVE_METRICS_PARAM not in self.parameters:
                raise ValueError(gettext('Parameter {} is required')
                                 .format(self.SAVE_METRICS_PARAM))

            import os.path
            map_storage = {'Defaul': 1,
                           'Stand Database': 4,
                           'File system': 5,
                           'HDFS Local': 6,
                           'Local File System': 7,
                           'path': '/srv/models'}

            # Hardcoded for a while - FIX_ME
            if self.save_storage == map_storage['Local File System']:
                self.import_code['callbacks'].append('ModelCheckpoint')

                if self.save_storage is None:
                    raise ValueError(gettext('Parameter {} is required.')
                                     .format(self.SAVE_STORAGE_PARAM))
                if self.save_name is None:
                    raise ValueError(gettext('Parameter {} is required.')
                                     .format(self.SAVE_NAME_PARAM))
                if self.save_action_if_exists is None:
                    raise ValueError(gettext('Parameter {} is required.')
                                     .format(self.SAVE_ACTION_IF_EXISTS_PARAM))
                if self.save_metrics is None:
                    raise ValueError(gettext('Parameter {} is required.')
                                     .format(self.SAVE_METRICS_PARAM))

                subset = []
                if self.save_subset.lower() == 'validation':
                    subset.append('val')
                elif self.save_subset.lower() == 'training':
                    subset.append('train')
                else:
                    subset.append('train')
                    subset.append('val')

                monitor = []
                for metric in self.save_metrics:
                    for sub in subset:
                        monitor.append(('{}_{}'.format(sub, metric['key']))
                                       .replace('train_', ''))

                if self.save_name.strip():
                    file_names = []
                    formats = []
                    for metric in monitor:
                        file_names.append('{}_{}'.format(self.save_name,
                                                         metric))
                        format = 'epoch_{epoch:02d}-' + metric + '_{' + \
                                 metric + ':.2f}.hdf5'
                        formats.append(format)

                    file_models = []
                    for i in range(0, len(file_names)):
                        file_name = '{0}/{1}.{2}'.format(
                            map_storage['path'],
                            self.save_name,
                            formats[i])

                        if self.save_action_if_exists == 'Raise error':
                            is_file = os.path.isfile(file_name)

                            if is_file:
                                raise ValueError(gettext('File {} exists.')
                                                 .format(self.save_name))

                        file_models.append(file_name)

                        # Create the ModelCheckpoints
                        mcp = ''
                        count = 0
                        mcp_var = ''
                        for f in file_models:
                            if mcp:
                                mcp += '\n'

                            mcp_var += 'modelcheckpoint_{0}_callback' \
                                           .format(monitor[count]) + ', '

                            mcp += str('modelcheckpoint_{monitor}_callback = ModelCheckpoint(\n' \
                                       '    filepath=str("{file}"),\n' \
                                       '    monitor="{monitor}",\n' \
                                       '    save_best_only=True,\n' \
                                       '    save_weights_only={save_weights_only},\n' \
                                       '    mode="auto",\n' \
                                       '    period=1)'.format(
                                file=f,
                                monitor=monitor[count],
                                save_weights_only=self.save_weights_only
                            ))
                            count += 1
                else:
                    raise ValueError(gettext('Parameter {} invalid.')
                                     .format(self.SAVE_NAME_PARAM))
            else:
                raise ValueError(gettext('Parameter {} not supported yet.')
                                 .format(self.SAVE_STORAGE_PARAM))

        # TO_DO ADD CALLBACKS CODE GENERATOR
        callbacks = '[JuicerCallback(emit_event, {}, "{}", {}), '.format(
            self.parameters['job_id'], self.parameters.get('task').get('id'),
            epochs)
        if self.callbacks is not None:
            for callback in self.callbacks:
                if self.callbacks:
                    self.callback_code += '\n'

                callbacks += str(callback['key'].lower()) + '_callback, '
                self.import_code['callbacks'].append(callback['key'])

                username = self.parameters['user']['name'].lower().split()[0:2]
                username = '_'.join(username)

                if callback['key'].lower() == 'tensorboard':
                    tb = 'tensorboard_callback = {callbak}(log_dir="/tmp/tensorboard/' \
                         '{user_id}_{username}/{workflow_id}_{job_id}")' \
                        .format(
                        user_id=self.parameters['workflow']['user']['id'],
                        workflow_id=self.parameters['workflow']['id'],
                        job_id=self.parameters['job_id'],
                        username=username,
                        callbak=self.import_code['callbacks'][-1])
                    self.callback_code += tb

                elif callback['key'].lower() == 'history':
                    ht = 'history_callback = {callbak}()' \
                        .format(callbak=self.import_code['callbacks'][-1])
                    self.callback_code += ht

            # Add the ModelCheckpoint code (mcp)
            if self.save_enabled:
                self.callback_code += '\n' + mcp
                callbacks += mcp_var

            callbacks += ']'
            callbacks = callbacks.replace(', ]', ']')

            self.callbacks = """callbacks={callbacks}""" \
                .format(callbacks=callbacks)
            functions_required_fit_generator.append(self.callbacks)

        if self.validation_generator is not None:
            self.validation_generator = """validation_data={}""".format(
                self.validation_generator)
            functions_required_fit_generator.append(self.validation_generator)

            if self.validation_steps is not None:
                self.validation_steps = int(self.validation_steps)
                functions_required_fit_generator.append(
                    """validation_steps={validation_steps}""".format(
                        validation_steps=self.validation_steps))
            else:
                self.validation_steps = self.steps_per_epoch
                functions_required_fit_generator.append(
                    """validation_steps={validation_steps}""".format(
                        validation_steps=self.validation_steps))

            if self.validation_freq is not None:
                self.validation_freq = get_int_or_tuple(self.validation_freq)
                if self.validation_freq is None:
                    self.validation_freq = string_to_list(self.validation_freq)

                if self.validation_freq is not None:
                    self.validation_freq = """validation_freq={validation_freq}
                    """.format(validation_freq=self.validation_freq)
                    functions_required_fit_generator.append(
                        self.validation_freq)
                else:
                    raise ValueError(gettext('Parameter {} is invalid.')
                                     .format(self.VALIDATION_FREQ_PARAM))

        if self.class_weight is not None:
            self.class_weight = string_to_dictionary(self.class_weight)
            if self.class_weight is not None:
                self.class_weight = """class_weight={class_weight}""" \
                    .format(class_weight=self.class_weight)
                functions_required_fit_generator.append(self.class_weight)
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.CLASS_WEIGHT_PARAM))

        if self.max_queue_size is not None:
            self.max_queue_size = int(self.max_queue_size)
            self.max_queue_size = """max_queue_size={max_queue_size}""" \
                .format(max_queue_size=self.max_queue_size)
            functions_required_fit_generator.append(self.max_queue_size)

        if self.workers < 0:
            raise ValueError(gettext('Parameter {} is invalid.')
                             .format(self.WORKERS_PARAM))
        if self.workers != 1:
            self.workers = int(self.workers)
            self.workers = """workers={workers}""" \
                .format(workers=self.workers)
            functions_required_fit_generator.append(self.workers)

        if self.use_multiprocessing:
            self.use_multiprocessing = """use_multiprocessing={}""".format(
                self.use_multiprocessing)
            functions_required_fit_generator.append(self.use_multiprocessing)

        if self.shuffle:
            self.shuffle = """shuffle={shuffle}""".format(shuffle=self.shuffle)
            functions_required_fit_generator.append(self.shuffle)

        if self.initial_epoch is not None:
            self.initial_epoch = int(self.initial_epoch)
            self.initial_epoch = """initial_epoch={initial_epoch}""" \
                .format(initial_epoch=self.initial_epoch)
            functions_required_fit_generator.append(self.initial_epoch)

        self.add_functions_required_fit_generator = ',\n    ' \
            .join(functions_required_fit_generator)

        self.classification_report = self.classification_report in ['1', 1]

        if self.classification_report:
            self.import_code['others'] = ["from sklearn.metrics import "
                                          "classification_report, "
                                          "confusion_matrix"]

    def generate_code(self):
        return dedent(
            """
            {callback_code}
            """
        ).format(callback_code=self.callback_code)

    def generate_history_code(self):
        if self.train_generator:
            if self.classification_report:
                if self.validation_generator:
                    if self.is_video_generator:
                        return dedent(
                            """
                            history = {var_name}.fit_generator(
                            {add_functions_required_fit_generator}
                            )
                            emit_event(
                                name='update task',
                                message=tab(table=history.history,
                                            add_epoch=True,
                                            metric='History',
                                            headers=list(history.history.keys())
                                ),
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            # Reports for the training
                            batch_size = training_video_generator.batch_size
                            number_of_videos = len(training_video_generator.classes)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {var_name}.predict_generator(
                                generator=predict_{train_generator},
                                steps=steps,
                                verbose=1
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            report = classification_report(
                                y_true=predict_training_video_generator.classes,
                                y_pred=predictions_to_matrix,
                                labels=list(class_mapping.values()),
                                target_names=list(class_mapping.keys()),
                                output_dict=False
                            )
                            
                            message = '\\n<h5>Classification Report - Training</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            ) 
                            
                            # Reports for the validation
                            batch_size = validation_video_generator.batch_size
                            number_of_videos = len(validation_video_generator.classes)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                            
                            predictions = {var_name}.predict_generator(
                                generator=predict_{val_generator},
                                steps=steps,
                                verbose=1
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            report = classification_report(
                                y_true=predict_validation_video_generator.classes,
                                y_pred=predictions_to_matrix,
                                labels=list(class_mapping.values()),
                                target_names=list(class_mapping.keys()),
                                output_dict=False
                            )
                            
                            message = '\\n<h5>Classification Report - Validation</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            """
                        ).format(var_name=self.var_name,
                                 add_functions_required_fit_generator=
                                 self.add_functions_required_fit_generator,
                                 val_generator=self.validation_generator
                                 .replace('validation_data=', ''),
                                 train_generator=self.train_generator
                                 .replace('generator=', '').replace(' ', ''),
                                 task_id=self.output_task_id)
                    else:
                        return dedent(
                            """
                            history = {var_name}.fit_generator(
                            {add_functions_required_fit_generator}
                            )
                            emit_event(
                                name='update task',
                                message=tab(table=history.history,
                                            add_epoch=True,
                                            metric='History',
                                            headers=list(history.history.keys())
                                ),
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            # Reports for the training
                            batch_size = {train_generator}.batch_size
                            number_of_videos = len({train_generator}.classes)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {var_name}.predict_generator(
                                generator={train_generator},
                                steps=steps
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            target_names = list({train_generator}.class_indices.keys())
                            labels = list({train_generator}.class_indices.values())
                            report = classification_report(
                                y_true={train_generator}.classes,
                                y_pred=predictions_to_matrix,
                                labels=labels,
                                target_names=target_names,
                                output_dict=False
                            )
                            
                            message = '\\n<h5>Classification Report - Training</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            ) 
                            
                            # Reports for the validation
                            batch_size = {val_generator}.batch_size
                            number_of_videos = len({val_generator}.classes)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {var_name}.predict_generator(
                                generator={val_generator},
                                steps=steps,
                                workers={workers},
                                use_multiprocessing=True
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            target_names = list({val_generator}.class_indices.keys())
                            labels = list({val_generator}.class_indices.values())
                            report = classification_report(
                                y_true={val_generator}.classes,
                                y_pred=predictions_to_matrix,
                                labels=labels,
                                target_names=target_names,
                                output_dict=False
                            )
                            
                            message = '\\n<h5>Classification Report - Validation</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            """
                        ).format(var_name=self.var_name,
                                 add_functions_required_fit_generator=
                                 self.add_functions_required_fit_generator,
                                 val_generator=self.validation_generator
                                 .replace('validation_data=', ''),
                                 train_generator=self.train_generator
                                 .replace('generator=', '').replace(' ', ''),
                                 task_id=self.output_task_id,
                                 workers=str(self.workers).replace('workers=', '')
                                 .replace(' ', ''))
                else:
                    if self.is_video_generator:
                        return dedent(
                            """
                            history = {var_name}.fit_generator(
                            {add_functions_required_fit_generator}
                            )
                            emit_event(
                                name='update task',
                                message=tab(table=history.history,
                                            add_epoch=True,
                                            metric='History',
                                            headers=list(history.history.keys())
                                ),
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            # Reports for the training
                            batch_size = training_video_generator.batch_size
                            number_of_videos = len(training_video_generator.classes)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {var_name}.predict_generator(
                                generator=predict_{train_generator},
                                steps=steps,
                                verbose=1
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            report = classification_report(
                                y_true=predict_training_video_generator.classes,
                                y_pred=predictions_to_matrix,
                                labels=list(class_mapping.values()),
                                target_names=list(class_mapping.keys()),
                                output_dict=False
                            )
                            
                            message = '\\n<h5>Classification Report - Training</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            """
                        ).format(var_name=self.var_name,
                                 add_functions_required_fit_generator=
                                 self.add_functions_required_fit_generator,
                                 val_generator=self.validation_generator
                                 .replace('validation_data=', ''),
                                 train_generator=self.train_generator
                                 .replace('generator=', '').replace(' ', ''),
                                 task_id=self.output_task_id)
                    else:
                        return dedent(
                            """
                            history = {var_name}.fit_generator(
                            {add_functions_required_fit_generator}
                            )
                            emit_event(
                                name='update task',
                                message=tab(table=history.history,
                                            add_epoch=True,
                                            metric='History',
                                            headers=list(history.history.keys())
                                ),
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            # Reports for the training
                            batch_size = {train_generator}.batch_size
                            number_of_videos = len({train_generator}.classes)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {var_name}.predict_generator(
                                generator={train_generator},
                                steps=steps
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            target_names = list({train_generator}.class_indices.keys())
                            labels = list({train_generator}.class_indices.values())
                            report = classification_report(
                                y_true={train_generator}.classes,
                                y_pred=predictions_to_matrix,
                                labels=labels,
                                target_names=target_names,
                                output_dict=False
                            )
                            
                            message = '\\n<h5>Classification Report - Training</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            ) 
                            
                            """
                        ).format(var_name=self.var_name,
                                 add_functions_required_fit_generator=
                                 self.add_functions_required_fit_generator,
                                 val_generator=self.validation_generator
                                 .replace('validation_data=', ''),
                                 train_generator=self.train_generator
                                 .replace('generator=', '').replace(' ', ''),
                                 task_id=self.output_task_id,
                                 workers=self.workers.replace('workers=', '')
                                 .replace(' ', ''))
            else:
                return dedent(
                    """
                    history = {var_name}.fit_generator(
                    {add_functions_required_fit_generator}
                    )
                    emit_event(name='update task',
                        message=tab(table=history.history, add_epoch=True, metric='History', headers=list(history.history.keys())),
                        type='HTML',
                        status='RESULTS',
                        identifier='{task_id}'
                    )
                    """
                ).format(var_name=self.var_name,
                         add_functions_required_fit_generator=
                         self.add_functions_required_fit_generator,
                         task_id=self.output_task_id)


class Model(Operation):
    # Compile
    OPTIMIZER_PARAM = 'optimizer'
    LOSS_PARAM = 'loss'
    METRICS_PARAM = 'metrics'
    K_PARAM = 'k'
    LOSS_WEIGHTS_PARAM = 'loss_weights'
    SAMPLE_WEIGHT_MODE_PARAM = 'sample_weight_mode'
    WEIGHTED_METRICS_PARAM = 'weighted_metrics'
    TARGET_TENSORS_PARAM = 'target_tensors'
    KWARGS_PARAM = 'kwargs'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        # Compile
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM, None)
        self.loss = parameters.get(self.LOSS_PARAM, None)
        self.metrics = parameters.get(self.METRICS_PARAM, None)
        self.k = parameters.get(self.K_PARAM, None)
        self.loss_weights = parameters.get(self.LOSS_WEIGHTS_PARAM,
                                           None)
        self.sample_weight_mode = parameters.get(self.SAMPLE_WEIGHT_MODE_PARAM,
                                                 None)
        self.weighted_metrics = parameters.get(self.WEIGHTED_METRICS_PARAM,
                                               None)
        self.target_tensors = parameters.get(self.TARGET_TENSORS_PARAM,
                                             None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.callback_code = ''

        self.task_name = self.parameters.get('task').get('name')
        self.parents = ""
        self.var_name = ""
        self.has_code = True
        self.show_results = True

        self.add_functions_required_compile = ""
        self.add_functions_required_fit_generator = ""

        self.output_task_id = self.parameters.get('task').get('id')

        if self.OPTIMIZER_PARAM not in parameters or self.optimizer is None:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.OPTIMIZER_PARAM))

        if self.LOSS_PARAM not in parameters or self.loss is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.LOSS_PARAM))

        if self.METRICS_PARAM not in parameters or self.metrics is None:
            raise ValueError(gettext('Parameter {} is required')
                             .format(self.METRICS_PARAM))

        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])

        if len(self.parents_by_port) == 0:
            raise ValueError(gettext('The operation needs the inputs.'))

        self.input_layers = []
        self.output_layers = []
        self.train_generator = None
        self.validation_generator = None

        self.is_video_generator = False

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': 'Model',
                            'preprocessing_image': None,
                            'others': None}

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        validation = ''
        for parent in self.parents_by_port:
            if str(parent[0]) == 'input layer':
                self.input_layers.append(convert_variable_name(parent[1]))
            elif str(parent[0]) == 'output layer':
                self.output_layers.append(convert_variable_name(parent[1]))
            elif str(parent[0]) == 'train-generator':
                self.train_generator = 'train_{var_name}'.format(
                    var_name=convert_variable_name(parent[1]))
                if validation == '':
                    validation = convert_variable_name(parent[1])
            elif str(parent[0]) == 'validation-generator':
                validation = convert_variable_name(parent[1])

        for i in range(len(self.input_layers)):
            if self.parents_slug[i] == 'model':
                self.parents[i] += '.output'

        if validation:
            self.validation_generator = 'validation_{var_name}'.format(
                var_name=validation)

        if self.train_generator is None:
            self.show_results = False
            if self.validation_generator:
                raise ValueError(gettext('It is not possible to use only '
                                         'validation data.'))

        if len(self.input_layers) == 0:
            raise ValueError(gettext('It is necessary to inform the input(s) '
                                     'layer(s).'))
        if len(self.output_layers) == 0:
            raise ValueError(gettext('It is necessary to inform the output(s) '
                                     'layer(s).'))

        input_layers_vector = '['
        for input_layer in self.input_layers:
            input_layers_vector += input_layer + ', '
        input_layers_vector += ']'
        self.input_layers = input_layers_vector.replace(', ]', ']')

        output_layers_vector = '['
        for output_layer in self.output_layers:
            output_layers_vector += output_layer + ', '
        output_layers_vector += ']'
        self.output_layers = output_layers_vector.replace(', ]', ']')

        # Compile
        functions_required_compile = []
        if self.optimizer is not None:
            self.optimizer = """    optimizer='{optimizer}'""" \
                .format(optimizer=self.optimizer)
            functions_required_compile.append(self.optimizer)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.optimizer))

        losses = []
        if self.loss is not None:
            for loss in self.loss:
                losses.append(loss['key'])

            self.loss = """loss={loss}""".format(loss=losses)
            functions_required_compile.append(self.loss)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.LOSS_PARAM))

        metrics = []
        if self.metrics is not None:
            for metric in self.metrics:
                metrics.append(str(metric['key']))

            self.metrics = """metrics={metrics}""" \
                .format(metrics=metrics)
            functions_required_compile.append(self.metrics)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.METRICS_PARAM))

        if 'sparse_top_k_categorical_accuracy' in metrics or \
                'top_k_categorical_accuracy' in metrics:
            self.k = """k={k}""" \
                .format(k=self.k)
            functions_required_compile.append(self.k)

        if self.loss_weights is not None:
            self.loss_weights = string_to_list(self.loss_weights)
            if self.loss_weights is None:
                self.loss_weights = string_to_dictionary(self.loss_weights)
                if self.loss_weights is None:
                    raise ValueError(gettext('Parameter {} is invalid.')
                                     .format(self.LOSS_WEIGHTS_PARAM))

            if self.loss_weights is not None:
                self.loss_weights = """loss_weights={loss_weights}""" \
                    .format(loss_weights=self.loss_weights)
                functions_required_compile.append(self.loss_weights)

        if self.sample_weight_mode is not None:
            if not self.sample_weight_mode == 'temporal':
                self.sample_weight_mode = string_to_list(
                    self.sample_weight_mode)
                if self.sample_weight_mode is None:
                    self.sample_weight_mode = string_to_dictionary(
                        self.sample_weight_mode)
                    if self.sample_weight_mode is None:
                        raise ValueError(gettext('Parameter {} is invalid.')
                                         .format(self.SAMPLE_WEIGHT_MODE_PARAM))

            self.sample_weight_mode = """sample_weight_mode=
            {sample_weight_mode}""" \
                .format(sample_weight_mode=self.sample_weight_mode)
            functions_required_compile.append(self.sample_weight_mode)

        if self.weighted_metrics is not None:
            self.weighted_metrics = string_to_list(self.weighted_metrics)
            if self.weighted_metrics is None:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.WEIGHTED_METRICS_PARAM))
            self.weighted_metrics = """weighted_metrics={weighted_metrics}""" \
                .format(weighted_metrics=self.weighted_metrics)
            functions_required_compile.append(self.weighted_metrics)

        if self.target_tensors is not None:
            self.target_tensors = """target_tensors={target_tensors}""" \
                .format(target_tensors=self.target_tensors)
            functions_required_compile.append(self.target_tensors)

        if self.kwargs is not None:
            self.kwargs = kwargs(self.kwargs)

            args = self.kwargs.split(',')
            args_params = self.kwargs.split('=')
            if len(args) >= 1 and ((len(args_params) - len(args)) == 1):
                self.kwargs = """{kwargs}""".format(kwargs=self.kwargs)
                functions_required_compile.append(self.kwargs)
            else:
                raise ValueError(gettext('Parameter {} is invalid.')
                                 .format(self.KWARGS_PARAM))

        self.add_functions_required_compile = ',\n    ' \
            .join(functions_required_compile)

    def generate_code(self):
        return dedent(
            """
            {callback_code}
            
            {var_name} = Model(
                inputs={inputs},
                outputs={outputs}
            )
            {var_name}.compile(
            {add_functions_required_compile}
            )

            summary_list_{var_name} = ['<h5>Summary</h5><pre>']
            summary = {var_name}.summary(
                print_fn=lambda x: summary_list_{var_name}.append(x))

            summary_list_{var_name}.append('</pre>')
            emit_event(name='update task',
                message='\\n'.join(summary_list_{var_name}),
                type='HTML',
                status='RESULTS',
                identifier='{task_id}')
            """
        ).format(var_name=self.var_name,
                 inputs=self.input_layers,
                 outputs=self.output_layers,
                 add_functions_required_compile=
                 self.add_functions_required_compile,
                 callback_code=self.callback_code,
                 task_id=self.output_task_id)
