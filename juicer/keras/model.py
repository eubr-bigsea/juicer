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
                        generator={generator},
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
                        generator={generator},
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
                        generator={generator},
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
                        generator={generator},
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

    ADVANCED_OPTIONS_PARAM = 'advanced_options'

    # Early stopping callback
    EARLY_STOPPING_PARAM = 'early_stopping'
    MONITOR_PARAM = 'monitor'
    MIN_DELTA_PARAM = 'min_delta'
    PATIENCE_PARAM = 'patience'
    MODE_PARAM = 'mode'
    BASELINE_PARAM = 'baseline'
    RESTORE_BEST_WEIGHTS_PARAM = 'restore_best_weights'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        # Fit Generator
        self.steps_per_epoch = parameters.get(self.STEPS_PER_EPOCH_PARAM, None)
        self.epochs = parameters.get(self.EPOCHS_PARAM, None)
        self.validation_data = parameters.get(self.VALIDATION_DATA_PARAM, None)
        self.validation_steps = parameters.get(self.VALIDATION_STEPS_PARAM,
                                               None)
        self.validation_freq = parameters.get(self.VALIDATION_FREQ_PARAM, None)
        self.class_weight = parameters.get(self.CLASS_WEIGHT_PARAM, None)
        self.max_queue_size = parameters.get(self.MAX_QUEUE_SIZE_PARAM, None)
        self.workers = int(parameters.get(self.WORKERS_PARAM, None))
        self.use_multiprocessing = parameters.get(
            self.USE_MULTIPROCESSING_PARAM, None)
        self.shuffle = parameters.get(self.SHUFFLE_PARAM, None)
        self.initial_epoch = parameters.get(self.INITIAL_EPOCH_PARAM, None)

        self.callbacks = [
            {'key': 'TensorBoard', 'value': 'TensorBoard'},
            {'value': 'History', 'key': 'History'},
            {'value': 'TimeLog', 'key': 'TimeLog'}
        ]

        # Save params
        self.save_enabled = parameters.get(self.SAVE_ENABLED_PARAM, None)
        self.save_storage = parameters.get(self.SAVE_STORAGE_PARAM, None)
        self.save_name = parameters.get(self.SAVE_NAME_PARAM, None)
        self.save_action_if_exists = parameters.get(
            self.SAVE_ACTION_IF_EXISTS_PARAM, None)
        self.save_weights_only = parameters.get(self.SAVE_WEIGHTS_ONLY_PARAM,
                                                None)
        self.save_metrics = parameters.get(self.SAVE_METRICS_PARAM, None)
        self.save_subset = parameters.get(self.SAVE_SUBSET_PARAM, None)

        self.classification_report = parameters.get(
            self.CLASSIFICATION_REPORT_PARAM, None)
        self.advanced_options = parameters.get(self.ADVANCED_OPTIONS_PARAM,
                                               None)

        self.early_stopping = int(parameters.get(self.EARLY_STOPPING_PARAM, 0))
        self.monitor = parameters.get(self.MONITOR_PARAM, None)
        self.min_delta = parameters.get(self.MIN_DELTA_PARAM, None)
        self.patience = parameters.get(self.PATIENCE_PARAM, None)
        self.mode = parameters.get(self.MODE_PARAM, None)
        self.baseline = parameters.get(self.BASELINE_PARAM, None)
        self.restore_best_weights = int(parameters.get(
            self.RESTORE_BEST_WEIGHTS_PARAM, 0))

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

        # if self.STEPS_PER_EPOCH_PARAM not in parameters or self.steps_per_epoch is None:
        #     raise ValueError(gettext('Parameter {} is required')
        #                      .format(self.STEPS_PER_EPOCH_PARAM))

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

        self.is_video_or_sequence_generator = False

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': 'Model',
                            'preprocessing_image': None,
                            'others': None}

        self.parents_by_port = parameters.get('my_ports', [])
        self.treatment()

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
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
                #self.var_name = self.model

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

        if 'video-generator' in self.parameters.get('parents_slug') or \
                'sequence-generator' in self.parameters.get('parents_slug'):
            self.is_video_or_sequence_generator = True
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
            self.train_generator = """    generator={train_generator}""".format(
                train_generator=self.train_generator)
            functions_required_fit_generator.append(self.train_generator)

        if self.steps_per_epoch is not None:
            functions_required_fit_generator.append(
                """steps_per_epoch={steps_per_epoch}""".format(
                    steps_per_epoch=self.steps_per_epoch))
        # else:
        #     raise ValueError(gettext('Parameter {} is required.')
        #                      .format(self.STEPS_PER_EPOCH_PARAM))

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
                           'Local File System': 7}

            if self.save_weights_only:
                map_storage['path'] = '/srv/weights'
            else:
                map_storage['path'] = '/srv/models'

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
                        # format = 'epoch_{epoch:02d}-' + metric + '_{' + \
                        #          metric + ':.2f}.hdf5'
                        # formats.append(format)

                    file_models = []
                    for i in range(0, len(file_names)):
                        file_name = '{0}/{1}.{2}'.format(
                            map_storage['path'],
                            file_names[i],
                            'hdf5')
                            #formats[i])

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

                        mcp_var += 'modelcheckpoint_{0}_callback, '.format(
                            monitor[count])

                        mcp += str('modelcheckpoint_{monitor}_callback = ModelCheckpoint(\n' \
                                   '    filepath=str("{file}"),\n' \
                                   '    monitor="{monitor}",\n' \
                                   '    save_best_only=True,\n' \
                                   '    save_weights_only={save_weights_only},\n' \
                                   '    mode="auto",\n' \
                                   '    period=1)'.format(
                            file=f,
                            monitor=monitor[count],
                            save_weights_only=self.save_weights_only))
                        count += 1
                else:
                    raise ValueError(gettext('Parameter {} invalid.')
                                     .format(self.SAVE_NAME_PARAM))
            else:
                raise ValueError(gettext('Parameter {} not supported yet.')
                                 .format(self.SAVE_STORAGE_PARAM))

        self.early_stopping = True if int(self.early_stopping) == 1 else \
            False
        self.restore_best_weights = True if int(self.restore_best_weights) == 1\
            else False

        if self.early_stopping:
            self.callbacks.append({'key': 'EarlyStopping',
                                   'value': 'EarlyStopping'})
            if self.baseline:
                self.baseline = float(self.baseline)
            if self.min_delta:
                self.min_delta = float(self.min_delta)
            if self.patience:
                self.patience = int(self.patience)

        # TO_DO ADD CALLBACKS CODE GENERATOR
        callbacks = '[\n\t\tJuicerCallback(emit_event, {}, "{}", {}),\n\t\t'.format(
            self.parameters['job_id'], self.parameters.get('task').get('id'),
            epochs)
        if self.callbacks is not None:
            for callback in self.callbacks:
                if self.callbacks:
                    self.callback_code += '\n'

                callbacks += str(callback['key'].lower()) + '_callback,\n\t\t'

                username = self.parameters['user']['name'].lower().split()[0:2]
                username = '_'.join(username)

                if callback['key'].lower() == 'tensorboard':
                    self.import_code['callbacks'].append(callback['key'])
                    tb = 'tensorboard_callback = {callback}(' \
                         '\n\tlog_dir="/tmp/tensorboard/{user_id}_{username}/{workflow_id}_{job_id}"' \
                         '\n)' \
                        .format(
                        user_id=self.parameters['workflow']['user']['id'],
                        workflow_id=self.parameters['workflow']['id'],
                        job_id=self.parameters['job_id'],
                        username=username,
                        callback=callback['key'])
                    self.callback_code += tb

                elif callback['key'].lower() == 'history':
                    self.import_code['callbacks'].append(callback['key'])
                    ht = '\nhistory_callback = {callbak}()' \
                        .format(callbak=callback['key'])
                    self.callback_code += ht

                elif callback['key'].lower() == 'earlystopping':
                    self.import_code['callbacks'].append(callback['key'])
                    es = '\nearlystopping_callback = {callbak}(' \
                         '\n\tmonitor="{monitor}",' \
                         '\n\tmin_delta={min_delta},' \
                         '\n\tpatience={patience},' \
                         '\n\tmode="{mode}",' \
                         '\n\tbaseline={baseline},' \
                         '\n\trestore_best_weights={restore_best_weights}' \
                         '\n)\n'.format(callbak=callback['key'],
                                    monitor=self.monitor,
                                    min_delta=self.min_delta,
                                    patience=self.patience,
                                    mode=self.mode,
                                    baseline=self.baseline,
                                    restore_best_weights=self.restore_best_weights)
                    self.callback_code += es

                elif callback['key'].lower() == 'timelog':
                    tl = '\ntimelog_callback = {callbak}(' \
                         '\n\tpath_to_save="{path_to_save}"' \
                         '\n)\n'.format(
                                    callbak=callback['key'],
                                    path_to_save="/srv/data/timelog")
                    self.callback_code += tl

            # Add the ModelCheckpoint code (mcp)
            if self.save_enabled:
                self.callback_code += '\n' + mcp
                callbacks += mcp_var

            callbacks += ']'
            callbacks = callbacks.replace(', ]', '\n\t]')

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
                    if self.is_video_or_sequence_generator:
                        return dedent(
                            """
                            history = {model}.fit_generator(
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
                            number_of_videos = len({train_generator}.videos_path)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {model}.predict_generator(
                                generator=predict_{train_generator},
                                steps=steps,
                                verbose=1
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            report = classification_report(
                                y_true=predict_{train_generator}.classes,
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
                            batch_size = {val_generator}.batch_size
                            number_of_videos = len({val_generator}.videos_path)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                            
                            predictions = {model}.predict_generator(
                                generator=predict_{val_generator},
                                steps=steps,
                                verbose=1
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            report = classification_report(
                                y_true=predict_{val_generator}.classes,
                                y_pred=predictions_to_matrix,
                                labels=list(class_mapping.values()),
                                target_names=list(class_mapping.keys()),
                                output_dict=False
                            )
                            
                            {var_name} = {model}
                            
                            message = '\\n<h5>Classification Report - Validation</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            """
                        ).format(model=self.model,
                                 var_name=self.var_name,
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
                            history = {model}.fit_generator(
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
                            number_of_videos = len({train_generator}.videos_path)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {model}.predict_generator(
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
                            number_of_videos = len({val_generator}.videos_path)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {model}.predict_generator(
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
                            
                            {var_name} = {model}
                            
                            message = '\\n<h5>Classification Report - Validation</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            """
                        ).format(model=self.model,
                                 var_name=self.var_name,
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
                    if self.is_video_or_sequence_generator:
                        return dedent(
                            """
                            history = {model}.fit_generator(
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
                            number_of_videos = len({train_generator}.videos_path)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {model}.predict_generator(
                                generator=predict_{train_generator},
                                steps=steps,
                                verbose=1
                            )
                            
                            predictions_to_matrix = np.argmax(predictions, axis=1)
                            
                            report = classification_report(
                                y_true=predict_{train_generator}.classes,
                                y_pred=predictions_to_matrix,
                                labels=list(class_mapping.values()),
                                target_names=list(class_mapping.keys()),
                                output_dict=False
                            )
                            
                            {var_name} = {model}
                            
                            message = '\\n<h5>Classification Report - Training</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            )
                            
                            """
                        ).format(model=self.model,
                                 var_name=self.var_name,
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
                            history = {model}.fit_generator(
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
                            number_of_videos = len({train_generator}.videos_path)
                            
                            if number_of_videos % batch_size == 0:
                                steps = number_of_videos // batch_size
                            else:
                                steps = (number_of_videos // batch_size) + 1
                                
                            predictions = {model}.predict_generator(
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
                            
                            {var_name} = {model}
                            
                            message = '\\n<h5>Classification Report - Training</h5>'
                            message += '<pre>' + report + '</pre>'
                            emit_event(name='update task',
                                message=message,
                                type='HTML',
                                status='RESULTS',
                                identifier='{task_id}'
                            ) 
                            
                            """
                        ).format(model=self.model,
                                 var_name=self.var_name,
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
                    history = {model}.fit_generator(
                    {add_functions_required_fit_generator}
                    )
                    
                    {var_name} = {model}
                    
                    emit_event(name='update task',
                        message=tab(table=history.history, add_epoch=True, metric='History', headers=list(history.history.keys())),
                        type='HTML',
                        status='RESULTS',
                        identifier='{task_id}'
                    )
                    """
                ).format(model=self.model,
                         var_name=self.var_name,
                         add_functions_required_fit_generator=
                         self.add_functions_required_fit_generator,
                         task_id=self.output_task_id)


class Load(Operation):
    # Compile
    MODEL_PARAM = 'model'
    WEIGHTS_PARAM = 'weights'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.model = parameters.get(self.MODEL_PARAM, None)
        self.weights = parameters.get(self.WEIGHTS_PARAM, None)

        self._model = None
        self._weights = None

        self.task_name = self.parameters.get('task').get('name')
        self.var_name = ""
        self.has_code = True
        self.parent = ""
        self.parents_by_port = parameters.get('my_ports', [])
        self.parents_slug = parameters.get('parents_slug', [])

        self.add_functions_required = ""

        self.output_task_id = self.parameters.get('task').get('id')

        if (self.model is None and self.weights is None) or \
                (self.model is not None and self.weights is not None):
            raise ValueError(gettext('It is necessary to inform {} or {}.')
                             .format(self.MODEL_PARAM, self.WEIGHTS_PARAM))

        supported_formats = ('HDF5',)

        if self.model != 0 and self.model is not None:
            self.metadata_model = self.get_data_source(
                data_source_id=self.model)

            if self.metadata_model.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported model format: {}').format(
                    self.metadata_model.get('format')))

            self.format = self.metadata_model.get('format')

            self.import_code = {'layer': None,
                                'callbacks': [],
                                'model': 'load_model',
                                'preprocessing_image': None,
                                'others': None}

        if self.weights != 0 and self.weights is not None:
            self.metadata_weights = self.get_data_source(
                data_source_id=self.weights)

            if self.metadata_weights.get('format') not in supported_formats:
                raise ValueError(gettext('Unsupported weights format: {}').format(
                    self.metadata_weights.get('format')))

            self.format = self.metadata_weights.get('format')

            self.import_code = {'layer': None,
                                'callbacks': [],
                                'model': '',
                                'preprocessing_image': None,
                                'others': None}
        self.treatment()

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
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        # for python_code in self.python_code_to_remove:
        #     self.parent.remove(python_code[0])

        if self.parent:
            self.parent = '{}'.format(self.parent[0])
        else:
            self.parent = ''

        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        if self.model is not None:
            self._model = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_model.get('storage').get('url'),
                file_url=self.metadata_model.get('url')
            )
            self._model = self._model.replace('file://', '')

        if self.weights is not None:
            if not self.parents_slug:
                raise ValueError(
                    gettext('You must connect a model to load the weights.'))
            # elif not self.parents_slug[0] == 'model':
            #     raise ValueError(
            #         gettext('You must enter a model to load the model.'))

            self._weights = """'{storage_url}/{file_url}'""".format(
                storage_url=self.metadata_weights.get('storage').get('url'),
                file_url=self.metadata_weights.get('url')
            )
            self._weights = self._weights.replace('file://', '')

    def generate_code(self):
        if self.model:
            return dedent(
                """                
                {var_name} = load_model({model})
                """.format(var_name=self.var_name, model=self._model)
            )
        elif self.weights:
            return dedent(
                """
                {var_name} = {parent}
                {var_name}.load_weights({weights})
                """.format(var_name=self.var_name,
                           weights=self._weights,
                           parent=self.parent)
            )


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

    #Advanced optimizer options
    ADVANCED_OPTIMIZER_PARAM = 'advanced_optimizer'
    CLIPNORM_PARAM = 'clipnorm'
    CLIPVALUE_PARAM = 'clipvalue'
    LEARNING_RATE_SGD_PARAM = 'learning_rate_sgd'
    DECAY_SGD_PARAM = 'decay_sgd'
    MOMENTUM_SGD_PARAM = 'momentum_sgd'
    NESTEROV_SGD_PARAM = 'nesterov_sgd'
    LEARNING_RATE_RMSPROP_PARAM = 'learning_rate_rmsprop'
    RHO_RMSPROP_PARAM = 'rho_rmsprop'
    LEARNING_RATE_ADAGRAD_PARAM = 'learning_rate_adagrad'
    LEARNING_RATE_ADADELTA_PARAM = 'learning_rate_adadelta'
    RHO_ADADELTA_PARAM = 'rho_adadelta'
    LEARNING_RATE_ADAM_PARAM = 'learning_rate_adam'
    BETA_1_ADAM_PARAM = 'beta_1_adam'
    BETA_2_ADAM_PARAM = 'beta_2_adam'
    AMSGRAD_ADAM_PARAM = 'amsgrad_adam'
    LEARNING_RATE_ADAMAX_PARAM = 'learning_rate_adamax'
    BETA_1_ADAMAX_PARAM = 'beta_1_adamax'
    BETA_2_ADAMAX_PARAM = 'beta_2_adamax'
    LEARNING_RATE_NADAM_PARAM = 'learning_rate_nadam'
    BETA_1_NADAM_PARAM = 'beta_1_nadam'
    BETA_2_NADAM_PARAM = 'beta_2_nadam'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        # Compile
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM, None)
        self.loss = parameters.get(self.LOSS_PARAM, None)
        self.metrics = parameters.get(self.METRICS_PARAM, None)
        self.k = parameters.get(self.K_PARAM, None)
        self.loss_weights = parameters.get(self.LOSS_WEIGHTS_PARAM, None)
        self.sample_weight_mode = parameters.get(self.SAMPLE_WEIGHT_MODE_PARAM,
                                                 None)
        self.weighted_metrics = parameters.get(self.WEIGHTED_METRICS_PARAM,
                                               None)
        self.target_tensors = parameters.get(self.TARGET_TENSORS_PARAM, None)
        self.kwargs = parameters.get(self.KWARGS_PARAM, None)

        self.advanced_optimizer = int(parameters.get(
            self.ADVANCED_OPTIMIZER_PARAM, None))
        self.clipnorm = float(parameters.get(self.CLIPNORM_PARAM, None))
        self.clipvalue = float(parameters.get(self.CLIPVALUE_PARAM, None))
        self.learning_rate_sgd = float(parameters.get(
            self.LEARNING_RATE_SGD_PARAM,None))
        self.decay_sgd = float(parameters.get(self.DECAY_SGD_PARAM, None))
        self.momentum_sgd = float(parameters.get(self.MOMENTUM_SGD_PARAM, None))
        self.nesterov_sgd = int(parameters.get(self.NESTEROV_SGD_PARAM, None))
        self.learning_rate_rmsprop = float(parameters.get(
            self.LEARNING_RATE_RMSPROP_PARAM, None))
        self.rho_rmsprop = float(parameters.get(self.RHO_RMSPROP_PARAM, None))
        self.learning_rate_adagrad = float(parameters.get(
            self.LEARNING_RATE_ADAGRAD_PARAM, None))
        self.learning_rate_adadelta = float(parameters.get(
            self.LEARNING_RATE_ADADELTA_PARAM, None))
        self.rho_adadelta = float(parameters.get(self.RHO_ADADELTA_PARAM, None))
        self.learning_rate_adam = float(parameters.get(
            self.LEARNING_RATE_ADAM_PARAM,None))
        self.beta_1_adam = float(parameters.get(self.BETA_1_ADAM_PARAM, None))
        self.beta_2_adam = float(parameters.get(self.BETA_2_ADAM_PARAM, None))
        self.amsgrad_adam = int(parameters.get(self.AMSGRAD_ADAM_PARAM, None))
        self.learning_rate_adamax = float(parameters.get(
            self.LEARNING_RATE_ADAMAX_PARAM, None))
        self.beta_1_adamax = float(parameters.get(self.BETA_1_ADAMAX_PARAM,
                                                  None))
        self.beta_2_adamax = float(parameters.get(self.BETA_2_ADAMAX_PARAM,
                                                  None))
        self.learning_rate_nadam = float(parameters.get(
            self.LEARNING_RATE_NADAM_PARAM, None))
        self.beta_1_nadam = float(parameters.get(self.BETA_1_NADAM_PARAM, None))
        self.beta_2_nadam = float(parameters.get(self.BETA_2_NADAM_PARAM, None))

        self._clipnorm = None
        self._clipvalue = None
        self._learning_rate_sgd = None
        self._decay_sgd = None
        self._momentum_sgd = None
        self._nesterov_sgd = None
        self._learning_rate_rmsprop = None
        self._rho_rmsprop = None
        self._learning_rate_adagrad = None
        self._learning_rate_adadelta = None
        self._rho_adadelta = None
        self._learning_rate_adam = None
        self._beta_1_adam = None
        self._beta_2_adam = None
        self._amsgrad_adam = None
        self._learning_rate_adamax = None
        self._beta_1_adamax = None
        self._beta_2_adamax = None
        self._learning_rate_nadam = None
        self._beta_1_nadam = None
        self._beta_2_nadam = None

        self.optimizer_function = None

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

        self.advanced_optimizer = True if int(self.advanced_optimizer) == 1 \
            else False
        self.nesterov_sgd = True if int(self.nesterov_sgd) == 1 else False
        self.amsgrad_adam = True if int(self.amsgrad_adam) == 1 else False

        functions_required_compile = []

        if self.optimizer.strip():
            if not self.advanced_optimizer:
                self.optimizer = """    optimizer='{optimizer}'""" \
                    .format(optimizer=self.optimizer)
                functions_required_compile.append(self.optimizer)

                self.optimizer_function = ""
            else:
                # All optimizers
                if 0.0 <= self.clipnorm <= 1.0:
                    self._clipnorm = "clipnorm={}".format(self.clipnorm)
                else:
                    raise ValueError(gettext('Inform a value between 0.0 and 1.'
                                             '0.').format(self.CLIPNORM_PARAM))
                if 0.0 <= self.clipvalue <= 1.0:
                    self._clipvalue = "clipvalue={}".format(self.clipvalue)
                else:
                    raise ValueError(gettext('Inform a value between 0.0 and 1.'
                                             '0.').format(self.CLIPVALUE_PARAM))

                # SGD optimizer
                if self.optimizer == 'sgd':
                    if self.learning_rate_sgd >= 0.0:
                        self._learning_rate_sgd = "lr={}".format(
                            self.learning_rate_sgd)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    if self.decay_sgd >= 0.0:
                        self._decay_sgd = "decay={}".format(self.decay_sgd)
                    else:
                        raise ValueError(gettext('{} - Inform a value >= 0.0.'
                                                 ).format('decay'))

                    if self.momentum_sgd >= 0.0:
                        self._momentum_sgd = "momentum={}".format(
                            self.momentum_sgd)
                    else:
                        raise ValueError(gettext('{} - Inform a value >= 0.0'
                                                 ).format('momentum'))

                    self._nesterov_sgd = "nesterov={}".format(self.nesterov_sgd)

                    self.import_code['others'] = ['from keras.optimizers ' \
                                                 'import SGD']

                    self.optimizer_function = dedent(
                        """sgd_optimizer = SGD(\n\t{},\n\t{},\n\t{},\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_sgd, self._decay_sgd,
                                    self._momentum_sgd, self._nesterov_sgd,
                                    self._clipnorm, self._clipvalue)

                    self.optimizer = """    optimizer=sgd_optimizer"""
                    functions_required_compile.append(self.optimizer)

                # RMSprop optimizer
                elif self.optimizer == 'rmsprop':
                    if self.learning_rate_rmsprop >= 0.0:
                        self._learning_rate_rmsprop = "lr={}".format(
                            self.learning_rate_sgd)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    if self.rho_rmsprop >= 0.0:
                        self._rho_rmsprop = "rho={}".format(self.rho_rmsprop)
                    else:
                        raise ValueError(gettext('{} - Inform a value >= 0.0.'
                                                 ).format('rho'))

                    self.import_code['others'] = ['from keras.optimizers ' \
                                                  'import RMSprop']

                    self.optimizer_function = dedent(
                        """rmsprop_optimizer = RMSprop(\n\t{},\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_rmsprop,
                                    self._rho_rmsprop, self._clipnorm,
                                    self._clipvalue)

                    self.optimizer = """    optimizer=rmsprop_optimizer"""
                    functions_required_compile.append(self.optimizer)

                # Adagrad optimizer
                elif self.optimizer == 'adagrad':
                    if self.learning_rate_adagrad >= 0.0:
                        self._learning_rate_adagrad = "lr={}".format(
                            self.learning_rate_adagrad)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    self.import_code['others'] = ['from keras.optimizers ' \
                                                  'import Adagrad']

                    self.optimizer_function = dedent(
                        """adagrad_optimizer = Adagrad(\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_adagrad,
                                    self._clipnorm, self._clipvalue)

                    self.optimizer = """    optimizer=adagrad_optimizer"""
                    functions_required_compile.append(self.optimizer)

                # Adadelta optimizer
                elif self.optimizer == 'adadelta':
                    if self.learning_rate_adadelta >= 0.0:
                        self._learning_rate_adadelta = "lr={}".format(
                            self.learning_rate_adadelta)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    if self.rho_adadelta >= 0.0:
                        self._rho_adadelta = "rho={}".format(self.rho_radadelta)
                    else:
                        raise ValueError(gettext('{} - Inform a value >= 0.0.'
                                                 ).format('rho'))

                    self.import_code['others'] = ['from keras.optimizers ' \
                                                  'import Adadelta']

                    self.optimizer_function = dedent(
                        """adadelta_optimizer = Adadelta(\n\t{},\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_adadelta,
                                    self._rho_adadelta, self._clipnorm,
                                    self._clipvalue)

                    self.optimizer = """    optimizer=adadelta_optimizer"""
                    functions_required_compile.append(self.optimizer)

                # Adam optimizer
                elif self.optimizer == 'adam':
                    if self.learning_rate_adam >= 0.0:
                        self._learning_rate_adam = "lr={}".format(
                            self.learning_rate_adam)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    if 0 < self.beta_1_adam < 1:
                        self._beta_1_adam = "beta_1={}".format(self.beta_1_adam)
                    else:
                        raise ValueError(gettext(' 0 < {} < 1. Generally '
                                                 'close to 1.').format('beta_1')
                                         )

                    if 0 < self.beta_2_adam < 1:
                        self._beta_2_adam = "beta_2={}".format(self.beta_2_adam)
                    else:
                        raise ValueError(gettext(' 0 < {} < 1. Generally '
                                                 'close to 1.').format('beta_2')
                                         )

                    self._amsgrad_adam = "amsgrad={}".format(self.amsgrad_adam)

                    self.import_code['others'] = ['from keras.optimizers '
                                                  'import Adam']

                    self.optimizer_function = dedent(
                        """adam_optimizer = Adam(\n\t{},\n\t{},\n\t{},\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_adam, self._beta_1_adam,
                                    self._beta_2_adam, self._amsgrad_adam,
                                    self._clipnorm, self._clipvalue)

                    self.optimizer = """    optimizer=adam_optimizer"""
                    functions_required_compile.append(self.optimizer)

                # Adamax optimizer
                elif self.optimizer == 'adamax':
                    if self.learning_rate_adamax >= 0.0:
                        self._learning_rate_adamax = "lr={}".format(
                            self.learning_rate_adamax)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    if 0 < self.beta_1_adamax < 1:
                        self._beta_1_adamax = "beta_1={}".format(
                            self.beta_1_adamax)
                    else:
                        raise ValueError(gettext(' 0 < {} < 1. Generally '
                                                 'close to 1.').format('beta_1')
                                         )

                    if 0 < self.beta_2_adamax < 1:
                        self._beta_2_adamax = "beta_2={}".format(
                            self.beta_2_adamax)
                    else:
                        raise ValueError(gettext(' 0 < {} < 1. Generally '
                                                 'close to 1.').format('beta_2')
                                         )

                    self.import_code['others'] = ['from keras.optimizers ' \
                                                  'import Adamax']

                    self.optimizer_function = dedent(
                        """adamax_optimizer = Adamax(\n\t{},\n\t{},\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_adamax,
                                    self._beta_1_adamax, self._beta_2_adamax,
                                    self._clipnorm, self._clipvalue)

                    self.optimizer = """    optimizer=adamax_optimizer"""
                    functions_required_compile.append(self.optimizer)

                # Nadam optimizer
                elif self.optimizer == 'nadam':
                    if self.learning_rate_nadam >= 0.0:
                        self._learning_rate_nadam = "lr={}".format(
                            self.learning_rate_nadam)
                    else:
                        raise ValueError(gettext('{} - Inform a value '
                                                 'between 0.0 and 1.0.'
                                                 ).format('learning_rate'))

                    if 0 < self.beta_1_nadam < 1:
                        self._beta_1_nadam = "beta_1={}".format(
                            self.beta_1_nadam)
                    else:
                        raise ValueError(gettext(' 0 < {} < 1. Generally '
                                                 'close to 1.').format('beta_1')
                                         )

                    if 0 < self.beta_2_nadam < 1:
                        self._beta_2_nadam = "beta_2={}".format(
                            self.beta_2_nadam)
                    else:
                        raise ValueError(gettext(' 0 < {} < 1. Generally '
                                                 'close to 1.').format('beta_2')
                                         )

                    self.import_code['others'] = ['from keras.optimizers '
                                                  'import Nadam']

                    self.optimizer_function = dedent(
                        """nadam_optimizer = Adamax(\n\t{},\n\t{},\n\t{},\n\t{},\n\t{}\n)
                        """).format(self._learning_rate_nadam,
                                    self._beta_1_nadam, self._beta_2_nadam,
                                    self._clipnorm, self._clipvalue)

                    self.optimizer = """    optimizer=nadam_optimizer"""
                    functions_required_compile.append(self.optimizer)
        else:
            raise ValueError(gettext('Parameter {} is required.')
                             .format(self.OPTIMIZER_PARAM))

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

        self.add_functions_required_compile = ',\n    ' .join(
            functions_required_compile)

    def generate_code(self):
        return dedent(
            """
            {callback_code}
            
            {var_name} = Model(
                inputs={inputs},
                outputs={outputs}
            )
            
            {optimizer_function}
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
                 task_id=self.output_task_id,
                 optimizer_function=self.optimizer_function)


class Predict(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.task_name = self.parameters.get('task').get('name')
        self.var_name = ""
        self.has_code = True

        self.task_id = self.parameters.get('task').get('id')

        self.model = None
        self.generator = None
        self.data_type = None

        self.is_video = False

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': 'Model',
                            'preprocessing_image': None,
                            'others': ["from sklearn.metrics import "
                                       "classification_report, "
                                       "confusion_matrix"]}

        self.parents_by_port = parameters.get('parents_by_port', [])
        self.treatment()

    def treatment(self):
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name

        for parent in self.parents_by_port:
            if str(parent[0]) == 'model':
                self.model = convert_variable_name(parent[1])
            elif str(parent[0]) == 'generator':
                self.generator = convert_variable_name(parent[1])
            elif str(parent[0]) == 'train-image':
                self.data_type = 'train'
            elif str(parent[0]) == 'validation-image':
                self.data_type = 'validation'
            elif str(parent[0]) == 'test-image':
                self.data_type = 'test'
            elif str(parent[0]) == 'train-video':
                self.data_type = 'train'
            elif str(parent[0]) == 'validation-video':
                self.data_type = 'validation'
            elif str(parent[0]) == 'test-video':
                self.data_type = 'test'

            if 'video' in str(parent[0]):
                self.is_video = True

        if self.data_type:
            self.generator = '{}_{}'.format(self.data_type, self.generator)
        else:
            raise ValueError(gettext('It is necessary to inform the data '
                                     'source.')
                             )

    def generate_code(self):
        if self.is_video:
            return dedent(
                """                
                # Reports for the {data_type}
                {generator}.to_fit = False
                batch_size = {generator}.batch_size
                number_of_examples = len({generator}.videos_path)
                
                if number_of_examples % batch_size == 0:
                    steps = number_of_examples // batch_size
                else:
                    steps = (number_of_examples // batch_size) + 1
                    
                {var_name} = {model}.predict_generator(
                    generator={generator},
                    steps=steps
                )
                
                predictions_to_matrix = np.argmax({var_name}, axis=1)
                
                target_names = list(class_mapping.keys())
                labels = list(class_mapping.values())
                
                report = classification_report(
                    y_true={generator}.classes,
                    y_pred=predictions_to_matrix,
                    labels=labels,
                    target_names=target_names,
                    output_dict=False
                )
                
                message = '\\n<h5>Classification Report - Test</h5>'
                message += '<pre>' + report + '</pre>'
                emit_event(name='update task',
                    message=message,
                    type='HTML',
                    status='RESULTS',
                    identifier='{task_id}'
                ) 
                
                """
            ).format(var_name=self.var_name,
                     model=self.model,
                     generator=self.generator,
                     task_id=self.task_id,
                     data_type=self.data_type)
        else:
            return dedent(
                """                
                # Reports for the {data_type}
                {generator}.to_fit = False
                batch_size = {generator}.batch_size
                number_of_examples = len({generator}.classes)
                
                if number_of_examples % batch_size == 0:
                    steps = number_of_examples // batch_size
                else:
                    steps = (number_of_examples // batch_size) + 1
                    
                {var_name} = {model}.predict_generator(
                    generator={generator},
                    steps=steps
                )
                
                predictions_to_matrix = np.argmax({var_name}, axis=1)
                
                target_names = list({generator}.class_indices.keys())
                labels = list({generator}.class_indices.values())
                
                report = classification_report(
                    y_true={generator}.classes,
                    y_pred=predictions_to_matrix,
                    labels=labels,
                    target_names=target_names,
                    output_dict=False
                )
                
                label_to_target = {{}}
                for i in range(len(labels)):
                    label_to_target[labels[i]] = target_names[i]
            
                final_pred = {{
                    'file_name': [],
                    'predicted': [],
                    'class': []
                }}
                for i, f in enumerate(test_image_generator.filenames):
                    final_pred['file_name'].append({generator}.filenames[i])
                    final_pred['predicted'].append(label_to_target[predictions_to_matrix[i]])
                    final_pred['class'].append(label_to_target[{generator}.classes[i]])
                
                message = '\\n<h5>Classification Report - Test</h5>'
                message += '<pre>' + report + '</pre>'
                emit_event(name='update task',
                    message=message,
                    type='HTML',
                    status='RESULTS',
                    identifier='{task_id}'
                )
                
                emit_event(name='update task',
                    message=tab(table=final_pred, 
                                add_epoch=False,
                                metric='Classification by instance',
                                headers=list(final_pred.keys()),
                                show_index=True),
                    type='HTML',
                    status='RESULTS',
                    identifier='{task_id}'
                )
                
                """
            ).format(var_name=self.var_name,
                     model=self.model,
                     generator=self.generator,
                     task_id=self.task_id,
                     data_type=self.data_type)


class Evaluate(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.task_name = self.parameters.get('task').get('name')
        self.var_name = ""
        self.has_code = True

        self.output_task_id = self.parameters.get('task').get('id')

        self.model = None
        self.generator = None

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

        for parent in self.parents_by_port:
            if str(parent[0]) == 'model':
                self.model = convert_variable_name(parent[1])
            elif str(parent[0]) == 'generator':
                self.generator = convert_variable_name(parent[1])

    def generate_code(self):
        return dedent(
            """
            {var_name} = {model}.evaluate_generator(
                generator={generator}
            )
            print({model}.metrics_names)
            print({var_name})
            """
        ).format(var_name=self.var_name,
                 model=self.model,
                 generator=self.generator)