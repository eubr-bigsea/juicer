from textwrap import dedent
import uuid
from jinja2 import Environment, BaseLoader
from juicer.operation import Operation
from juicer.service import limonero_service

from juicer import auditing

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urllib.parse import urlparse, parse_qs
    from urllib.request import urlopen

import string


class SafeDict(dict):
    # noinspection PyMethodMayBeStatic
    def __missing__(self, key):
        return '{' + key + '}'


class ApplyModelOperation(Operation):
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = len(named_inputs) == 2 and any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.output = self.named_outputs.get('output data',
                                                 'out_data_{}'.format(self.order))

            self.model = self.named_inputs.get(
                'model', 'model_{}'.format(self.order))

            self.prediction = self.parameters.get(self.PREDICTION_PARAM, self.PREDICTION_PARAM)

            if self.FEATURES_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.__class__.__name__))
            self.features = self.parameters.get(self.FEATURES_PARAM)

        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        code = dedent("""
            {out} = {in1}.copy()
            X_train = {in1}[{features}].to_numpy().tolist()
            if hasattr({in2}, 'predict'):
                {out}['{new_attr}'] = {in2}.predict(X_train).tolist()
            else:
                # to handle scaler operations
                {out}['{new_attr}'] = {in2}.transform(X_train).tolist()
            """.format(out=self.output, in1=self.named_inputs['input data'], in2=self.model,
                       new_attr=self.prediction, features=self.features))

        return dedent(code)


class LoadModel(Operation):
    """LoadModel.

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'name' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('name', self.__class__))

        self.filename = parameters['name']
        self.output = named_outputs.get('output data',
                                        'output_data_{}'.format(self.order))

        self.has_code = len(named_outputs) > 0
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('output data', self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
        import pickle
        filename = '{filename}'
        {model} = pickle.load(open(filename, 'rb'))
        """.format(model=self.output, filename=self.filename)
        return dedent(code)


class SaveModel(Operation):
    """SaveModel.
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_PARAM = 'storage'
    SAVE_CRITERIA_PARAM = 'save_criteria'
    WRITE_MODE_PARAM = 'write_mode'

    CRITERIA_BEST = 'BEST'
    CRITERIA_ALL = 'ALL'
    CRITERIA_OPTIONS = [CRITERIA_BEST, CRITERIA_ALL]

    WRITE_MODE_ERROR = 'ERROR'
    WRITE_MODE_OVERWRITE = 'OVERWRITE'
    WRITE_MODE_OPTIONS = [WRITE_MODE_ERROR,
                          WRITE_MODE_OVERWRITE]

    WORKFLOW_NAME_PARAM = 'workflow_name'
    JOB_ID_PARAM = 'job_id'
    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.parameters = parameters

        self.name = parameters.get(self.NAME_PARAM)
        self.storage_id = parameters.get(self.STORAGE_PARAM)

        if self.name is None or self.storage_id is None:
            msg = _('Missing parameters. Check if values for parameters {} '
                    'were informed')
            raise ValueError(
                msg.format(', '.join([self.NAME_PARAM, self.STORAGE_PARAM])))

        self.path = parameters.get(self.PATH_PARAM, '/limonero/models').rstrip('/')

        self.write_mode = parameters.get(self.WRITE_MODE_PARAM, self.WRITE_MODE_ERROR)
        if self.write_mode not in self.WRITE_MODE_OPTIONS:
            raise ValueError(
                _('Invalid value for parameter {param}: value').format(
                    param=self.WRITE_MODE_PARAM, value=self.write_mode))

        self.criteria = parameters.get(self.SAVE_CRITERIA_PARAM, self.CRITERIA_ALL)
        if self.criteria not in self.CRITERIA_OPTIONS:
            raise ValueError(
                _('Invalid value for parameter {param}: {value}').format(
                    param=self.SAVE_CRITERIA_PARAM, value=self.criteria))

        self.filename = parameters.get(self.NAME_PARAM)
        if self.NAME_PARAM not in self.parameters:
            msg = _("Parameters '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.NAME_PARAM, self.__class__))

        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)
        self.workflow_name = parameters.get(self.WORKFLOW_NAME_PARAM)
        self.job_id = parameters.get(self.JOB_ID_PARAM)

        self.has_code = any([len(named_inputs) > 0, self.contains_results()])

    def get_audit_events(self):
        return [auditing.SAVE_MODEL]

    def generate_code(self):
        limonero_config = self.parameters.get('configuration') \
            .get('juicer').get('services').get('limonero')

        url = '{}'.format(limonero_config['url'], self.write_mode)
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        if storage['type'] != 'HDFS':
            raise ValueError(_('Storage type not supported: {}').format(
                storage['type']))

        if storage['url'].endswith('/'):
            storage['url'] = storage['url'][:-1]

        parsed = urlparse(storage['url'])
        models = self.named_inputs['models']
        if not isinstance(models, list):
            models = [models]

        user = self.parameters.get('user', {})
        code = dedent("""
            from juicer.scikit_learn.model_operation import ModelsEvaluationResultList
            from juicer.service.limonero_service import register_model

            all_models = [{models}]
            criteria = '{criteria}'
            if criteria == 'ALL':
                models_to_save = list(itertools.chain.from_iterable(
                    map(lambda m: m.models if isinstance(m,
                        ModelsEvaluationResultList) else [m], all_models)))
            elif criteria == 'BEST':
                raise ValueError('{msg2}')

            import pickle
            from io import BytesIO
            fs = pa.hdfs.connect('{hdfs_server}', {hdfs_port})
            
            def _save_model(model_to_save, model_path, model_name):
                final_model_path = '{final_url}/{{}}'.format(model_path)
                overwrite = '{overwrite}'
                exists = fs.exists(final_model_path)
 
                if exists:
                    if overwrite == 'OVERWRITE':
                        fs.delete(final_model_path, False)
                    else:
                        raise ValueError('{error_file_exists}')
                        
                with fs.open(final_model_path, 'wb') as f:
                    b = BytesIO()
                    pickle.dump(model_to_save, b)
                    f.write(b.getvalue())
                                
                # Save model information in Limonero
                model_type = '{{}}.{{}}'.format(model_to_save.__module__,
                    model_to_save.__class__.__name__)
                
                model_payload = {{
                    "user_id": {user_id},
                    "user_name": '{user_name}',
                    "user_login": '{user_login}',
                    "name": model_name,
                    "class_name": model_type,
                    "storage_id": {storage_id},
                    "path":  model_path,
                    "type": "UNSPECIFIED",
                    "task_id": '{task_id}',
                    "job_id": {job_id},
                    "workflow_id": {workflow_id},
                    "workflow_name": '{workflow_name}'
                }}
                # Save model information in Limonero
                register_model('{url}', model_payload, '{token}')

            for i, model in enumerate(models_to_save):
                if isinstance(model, dict): # For instance, it's a Indexer
                    for k, v in model.items():
                        name = '{name} - {{}}'.format(k)
                        path = '{path}/{name}.{{0}}.{{1:04d}}'.format(k, i)
                        _save_model(v, path, name)
                else:
                    name = '{name}'
                    path = '{path}/{name}.{{0:04d}}'.format(i)
                    _save_model(model, path, name)
        """.format(models=', '.join(models),
                   overwrite=self.write_mode,
                   path=self.path,
                   final_url=storage['url'],
                   url=url,
                   token=token,
                   storage_id=self.storage_id,
                   name=self.name.replace(' ', '_'),
                   criteria=self.criteria,
                   msg0=_('You cannot mix models with and without '
                          'evaluation (e.g. indexers) when saving models '
                          'and criteria is different from ALL'),
                   msg1=_('You cannot mix models built using with '
                          'different metrics ({}).'),
                   msg2=_('Invalid criteria.'),
                   error_file_exists=_('Model already exists'),
                   job_id=self.job_id,
                   task_id=self.parameters['task_id'],
                   workflow_id=self.workflow_id,
                   workflow_name=self.workflow_name,
                   user_id=user.get('id'),
                   user_name=user.get('name'),
                   user_login=user.get('login'),
                   hdfs_server=parsed.hostname,
                   hdfs_port=parsed.port,
                   ))
        return code

    def to_deploy_format(self, id_mapping):
        return []


class EvaluateModelOperation(Operation):
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    FEATURE_ATTRIBUTE_PARAM = 'feature'
    METRIC_PARAM = 'model_type'

    METRIC_TO_EVALUATOR = [
        'classification',
        'regression',
        'clustering'
    ]

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.prediction_attribute = (parameters.get(
            self.PREDICTION_ATTRIBUTE_PARAM) or [''])[0]
        # self.feature_attribute = (parameters.get(
        #         self.FEATURE_ATTRIBUTE_PARAM) or [''])[0]
        self.label_attribute = (parameters.get(
            self.LABEL_ATTRIBUTE_PARAM) or [''])[0]
        self.type_model = parameters.get(self.METRIC_PARAM) or ''

        if any([self.prediction_attribute == '', self.type_model == '']):
            msg = \
                _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.PREDICTION_ATTRIBUTE_PARAM,
                self.METRIC_PARAM, self.__class__))

        if self.type_model not in self.METRIC_TO_EVALUATOR:
            raise ValueError(_('Invalid metric value {}').format(
                self.type_model))

        # if self.type_model == 'clustering':
        #     if self.feature_attribute == '':
        #         msg = \
        #             _("Parameters '{}' must be informed for task {}")
        #         raise ValueError(msg.format(
        #                 self.FEATURE_ATTRIBUTE_PARAM, self.__class__))
        #     else:
        #         self.label_attribute = self.feature_attribute
        # else:
        if self.label_attribute == '':
            msg = \
                _("Parameters '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.LABEL_ATTRIBUTE_PARAM, self.__class__))

        self.has_code = any([(
                (len(self.named_inputs) > 0 and len(self.named_outputs) > 0) or
                (self.named_outputs.get('evaluator') is not None) or
                ('input data' in self.named_inputs)
        ), self.contains_results()])

        self.model = self.named_inputs.get(
            'model', 'model_{}'.format(self.order))

        self.model_out = self.named_outputs.get(
            'evaluated model', 'model_task_{}'.format(self.order))

        self.evaluator_out = self.named_outputs.get(
            'evaluator', 'evaluator_task_{}'.format(self.order))
        if not self.has_code and self.named_outputs.get(
                'evaluated model') is not None:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

        self.supports_cache = False
        self.has_import = "from sklearn.metrics import * \n"

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return ''

    def generate_code(self):
        if not self.has_code:
            return ''
        else:
            display_text = self.parameters['task']['forms'].get(
                'display_text', {'value': 1}).get('value', 1) in (1, '1')
            display_image = self.parameters['task']['forms'].get(
                'display_image', {'value': 1}).get('value', 1) in (1, '1')

            code = [dedent("""
                metric_value = 0.0
                display_text = {display_text}
                display_image = {display_image}
                model_type = '{model_type}'
                label_col = '{label_attr}'
                prediction_col = '{prediction_attr}'
                {model_output} = None

                y_pred = {input}[prediction_col].to_numpy().tolist()
                y_true = {input}[label_col].to_numpy().tolist()

                # Code for evaluating if the Label attribute is categorical
                from pandas.api.types import is_numeric_dtype
                if not is_numeric_dtype({input}[label_col]):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    le.fit(y_true)
                    final_label = le.transform(y_true)
                    if model_type != 'clustering':
                        final_pred = le.transform(y_pred)
                    else:
                        final_pred = y_pred
                else:
                    final_label = y_true
                    final_pred = y_pred
                # Used in summary              
                """)]

            if self.type_model == 'classification':
                self._get_code_for_classification_metrics(code)
            elif self.type_model == 'regression':
                self._get_code_for_regression_metrics(code)
            elif self.type_model == 'clustering':
                self._get_code_for_clustering_metrics(code)

            self._get_code_for_summary(code)

            code = "\n".join(code).format(
                display_text=display_text,
                display_image=display_image,
                evaluator_out=self.evaluator_out,
                join_plot_title=_('Prediction versus Residual'),
                join_plot_y_title=_('Residual'),
                join_plot_x_title=_('Prediction'),
                input=self.named_inputs['input data'],
                label_attr=self.label_attribute,
                model=self.model,
                model_output=self.model_out,
                model_type=self.type_model,
                operation_id=self.parameters['operation_id'],
                params_title=_('Parameters for this estimator'),
                params_table_headers=[_('Parameters'), _('Value')],
                plot_title=_('Actual versus Prediction'),
                plot_x_title=_('Actual'),
                plot_y_title=_('Prediction'),
                prediction_attr=self.prediction_attribute,
                table_headers=[_('Metric'), _('Value')],
                task_id=self.parameters['task_id'],
                title=_('Evaluation result'),
            )

            # Common for all metrics!
            # code += dedent("""
            #     {model_output} = ModelsEvaluationResultList(
            #             [{model}], {model}, '{metric}', metric_value)
            #
            #     {metric} = metric_value
            #     {model_output} = None
            #     """.format(
            #     model_output=self.model_out,
            #     model=self.model,
            #     metric=self.metric)

            return dedent(code)

    @staticmethod
    def _get_code_for_classification_metrics(code):
        """
        Generate code for other classification metrics besides those related to
        area.
        """
        code.append(dedent("""
            # classification metrics
            if display_image:
                # Test if feature indexer is in global cache, because
                # strings must be converted into numbers in order tho
                # run algorithms, but they are cooler when displaying
                # results.
                indexer = cached_state.get('indexers', {{}}).get(label_col)
                if indexer:
                    classes = indexer.labels
                else:
                    classes = sorted(list(set(y_true)))

                content = ConfusionMatrixImageReport(
                    cm=confusion_matrix(y_true, y_pred), classes=classes,)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='IMAGE', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})

            if display_text:

                headers = {table_headers}
                rows = [
                    ['F1', f1_score(y_true, y_pred, average='weighted')],
                    ['Weighted Precision', 
                     precision_score(y_true, y_pred, average='weighted')],
                    ['Weighted Recall', 
                     recall_score(y_true, y_pred, average='weighted')],
                    ['Accuracy', accuracy_score(y_true, y_pred)],
                    ['Cohens kappa', cohen_kappa_score(y_true, y_pred)],
                    ['Jaccard coefficient score', 
                     jaccard_score(y_true, y_pred, average='weighted')],
                    ['Matthews correlation coefficient (MCC)', 
                     matthews_corrcoef(y_true, y_pred)],
                ]

                if len(list(set(y_true))) == 2:
                    if set(y_true) != set([0,1]):
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        le.fit(y_true)
                        y_true = le.transform(y_true)
                        y_pred = le.transform(y_pred)

                    rows.append(['Area under ROC', 
                     roc_auc_score(y_true, y_pred)])
                    rows.append(['Area under Precision-Recall', 
                     average_precision_score(y_true, y_pred)])

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows,
                        title='{title}')

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """))

    @staticmethod
    def _get_code_for_clustering_metrics(code):
        """
        Code for the evaluator when metric is related to clustering
        """
        code.append(dedent("""
            # clustering metrics                        
            final_pred = np.array(final_pred).reshape(-1, 1)
            final_label = np.array(final_label).reshape(-1, 1)

            if display_text:
                headers = {table_headers}
                rows = [
                    ['Silhouette Coefficient',silhouette_score(final_label, final_pred)],
                    ['Calinski and Harabaz score', 
                     calinski_harabaz_score(final_label, final_pred)],
                ]

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows, title='{title}')

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """))

    @staticmethod
    def _get_code_for_regression_metrics(code):
        """
        Code for the evaluator when metric is related to regression
        """

        code.append(dedent("""
            # regression metrics
            if display_text:
                headers = {table_headers}
                rows = [
                    ['Mean squared error', mean_squared_error(y_true, y_pred)],
                    ['Root mean squared error', 
                     np.sqrt(mean_squared_error(y_true, y_pred))],
                    ['Mean absolute error', 
                     mean_absolute_error(y_true, y_pred)],
                    ['R^2 (coefficient of determination)', 
                     r2_score(y_true, y_pred)],
                ]

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows, title='{title}')

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})


            if len(y_true) < 2000 and display_image:
                residuals = [t - p for t, p in zip(y_true, y_pred)]
                pandas_df = pd.DataFrame.from_records(
                    [
                        dict(prediction=x[0], residual=x[1])
                            for x in zip(y_pred, residuals)
                    ]
                )

                report = SeabornChartReport()
                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=report.jointplot(pandas_df, 'prediction',
                        'residual', '{join_plot_title}',
                        '{join_plot_x_title}', '{join_plot_y_title}'),
                    type='IMAGE', title='{join_plot_title}',
                    task=dict(id='{task_id}'),
                    operation=dict(id={operation_id}),
                    operation_id={operation_id})

        """))

    @staticmethod
    def _get_code_for_summary(code):
        """
        Return code for model's summary (test if it is present)
        """
        code.append(dedent("""                               
            # model's summary       
            if len(y_true) < 2000 and display_image:

                report2 = MatplotlibChartReport()

                identity = range(int(max(final_label[-1], final_pred[-1])))
                emit_event(
                     'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=report2.plot(
                        '{plot_title}',
                        '{plot_x_title}',
                        '{plot_y_title}',
                        identity, identity, 'r.',
                        final_label, final_pred,'b.',),
                    type='IMAGE', title='{join_plot_title}',
                    task=dict(id='{task_id}'),
                    operation=dict(id={operation_id}),
                    operation_id={operation_id})

            if display_text:
                rows = []
                headers = {params_table_headers}
                params = {model}.get_params()
                for p in params:
                    rows.append([p, params[p]])

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows,
                        title='{params_title}')

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """))


class ModelsEvaluationResultList:
    """ Stores a list of ModelEvaluationResult """

    def __init__(self, models, best, metric_name, metric_value):
        self.models = models
        self.best = best
        self.metric_name = metric_name
        self.metric_value = metric_value


class CrossValidationOperation(Operation):
    """
    Cross validation operation used to evaluate classifier results using as many
    as folds provided in input.
    """
    NUM_FOLDS_PARAM = 'folds'
    EVALUATOR_PARAM = 'evaluator'
    SEED_PARAM = 'seed'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    FEATURE_ATTRIBUTE_PARAM = 'features'

    METRIC_TO_EVALUATOR = {
        'areaUnderROC': (
            'roc_auc', 'rawPredictionCol'),
        'areaUnderPR': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'f1': ('f1', 'predictionCol'),
        'weightedPrecision': (
            'precision_weighted', 'predictionCol'),
        'weightedRecall': (
            'recall_weighted', 'predictionCol'),
        'accuracy': (
            'accuracy', 'predictionCol'),
        'rmse': ('r2', 'predictionCol'),
        'mse': ('neg_mean_squared_error', 'predictionCol'),
        'mae': ('neg_mean_absolute_error', 'predictionCol'),
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(self.named_inputs) == 2])

        if self.EVALUATOR_PARAM not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.EVALUATOR_PARAM, self.__class__))

        self.metric = parameters.get(self.EVALUATOR_PARAM)

        if self.metric in self.METRIC_TO_EVALUATOR:
            self.evaluator = self.METRIC_TO_EVALUATOR[self.metric][0]
            self.param_prediction_arg = \
                self.METRIC_TO_EVALUATOR[self.metric][1]
        else:
            raise ValueError(
                _('Invalid metric value {}').format(self.metric))

        self.prediction_attr = parameters.get(
            self.PREDICTION_ATTRIBUTE_PARAM,
            'prediction')

        for param in [self.FEATURE_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM]:
            if param not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.LABEL_ATTRIBUTE_PARAM, self.__class__))

        self.label_attr = parameters.get(self.LABEL_ATTRIBUTE_PARAM)
        self.feature_attr = parameters.get(self.FEATURE_ATTRIBUTE_PARAM)

        self.num_folds = parameters.get(self.NUM_FOLDS_PARAM, 3)
        self.seed = parameters.get(self.SEED_PARAM, 'None')

        self.output = self.named_outputs.get(
            'scored data', 'scored_data_task_{}'.format(self.order))
        self.has_models = 'models' in self.named_outputs
        self.models = self.named_outputs.get(
            'models', 'models_task_{}'.format(self.order))
        self.best_model = self.named_outputs.get(
            'best model', 'best_model_{}'.format(self.order))

        self.algorithm_port = self.named_inputs.get(
            'algorithm', 'algo_{}'.format(self.order))
        self.input_port = self.named_inputs.get(
            'input data', 'in_{}'.format(self.order))
        self.evaluator_port = self.named_inputs.get(
            'evaluator', 'eval_{}'.format(self.order))

        self.has_import = \
            "from sklearn.model_selection import cross_val_score, KFold\n"


    @property
    def get_inputs_names(self):
        return ', '.join(
            [self.algorithm_port, self.input_port, self.evaluator_port])

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.best_model, self.models])

    def get_data_out_names(self, sep=','):
        return sep.join([self.output])

    def generate_code(self):
        code = dedent("""
                  kf = KFold(n_splits={folds}, random_state={seed}, 
                  shuffle=True)
                  X_train = {input_data}['{feature_attr}'].values
                  y = {input_data}['{label_attr}'].values

                  scores = cross_val_score({algorithm}, X_train.tolist(), 
                                           y.tolist(), cv=kf, scoring='{metric}')

                  best_score = np.argmax(scores)
                  """.format(algorithm=self.algorithm_port,
                             input_data=self.input_port,
                             evaluator=self.evaluator,
                             output=self.output,
                             best_model=self.best_model,
                             models=self.models,
                             prediction_arg=self.param_prediction_arg,
                             prediction_attr=self.prediction_attr,
                             label_attr=self.label_attr[0],
                             feature_attr=self.feature_attr[0],
                             folds=self.num_folds,
                             metric=self.metric,
                             seed=self.seed))

        if self.has_models:

            code += dedent("""
                    models = []
                    for train_index, test_index in kf.split(X_train):
                        Xf_train, Xf_test = X_train[train_index], X_train[test_index]
                        yf_train, yf_test = y[train_index],  y[test_index]
                        {algorithm}.fit(Xf_train.tolist(), yf_train.tolist())
                        models.append({algorithm})
                    
                    {best_model} = models[best_score]
                    """.format(algorithm=self.algorithm_port,
                               input_data=self.input_port,
                               evaluator=self.evaluator,
                               output=self.output,
                               best_model=self.best_model,
                               models=self.models,
                               prediction_arg=self.param_prediction_arg,
                               prediction_attr=self.prediction_attr,
                               label_attr=self.label_attr[0],
                               feature_attr=self.feature_attr[0],
                               folds=self.num_folds,
                               metric=self.metric,
                               seed=self.seed))
        else:
            code += dedent("""
                    models = None
                    train_index, test_index = list(kf.split(X_train))[best_score]
                    Xf_train, Xf_test = X_train[train_index], X_train[test_index]
                    yf_train, yf_test = y[train_index],  y[test_index]
                    {best_model} = {algorithm}.fit(Xf_train.tolist(), yf_train.tolist())
                    """.format(algorithm=self.algorithm_port,
                               input_data=self.input_port,
                               evaluator=self.evaluator,
                               output=self.output,
                               best_model=self.best_model,
                               models=self.models,
                               prediction_arg=self.param_prediction_arg,
                               prediction_attr=self.prediction_attr,
                               label_attr=self.label_attr[0],
                               feature_attr=self.feature_attr[0],
                               folds=self.num_folds,
                               metric=self.metric,
                               seed=self.seed))

        code += dedent("""
                metric_result = scores[best_score]
                {output} = {input_data}.copy()
                {output}['{prediction_attr}'] = {best_model}.predict(X_train.tolist())
                {models} = models
                """.format(algorithm=self.algorithm_port,
                           input_data=self.input_port,
                           evaluator=self.evaluator,
                           output=self.output,
                           best_model=self.best_model,
                           models=self.models,
                           prediction_arg=self.param_prediction_arg,
                           prediction_attr=self.prediction_attr,
                           label_attr=self.label_attr[0],
                           feature_attr=self.feature_attr[0],
                           folds=self.num_folds,
                           metric=self.metric,
                           seed=self.seed))

        # # If there is an output needing the evaluation result, it must be
        # # processed here (summarization of data results)
        # needs_evaluation = 'evaluation' in self.named_outputs and False
        # if needs_evaluation:
        #     eval_code = """
        #         grouped_result = fit_data.select(
        #                 evaluator.getLabelCol(), evaluator.getPredictionCol())\\
        #                 .groupBy(evaluator.getLabelCol(),
        #                          evaluator.getPredictionCol()).count().collect()
        #         eval_{output} = {{
        #             'metric': {{
        #                 'name': evaluator.getMetricName(),
        #                 'value': metric_result
        #             }},
        #             'estimator': {{
        #                 'name': estimator.__class__.__name__,
        #                 'predictionCol': evaluator.getPredictionCol(),
        #                 'labelCol': evaluator.getLabelCol()
        #             }},
        #             'confusion_matrix': {{
        #                 'data': json.dumps(grouped_result)
        #             }},
        #             'evaluator': evaluator
        #         }}
        #
        #         emit_event('task result', status='COMPLETED',
        #             identifier='{task_id}', message='Result generated',
        #             type='TEXT', title='{title}',
        #             task={{'id': '{task_id}' }},
        #             operation={{'id': {operation_id} }},
        #             operation_id={operation_id},
        #             content=json.dumps(eval_{output}))
        #
        #         """.format(output=self.output,
        #                    title='Evaluation result',
        #                    task_id=self.parameters['task_id'],
        #                    operation_id=self.parameters['operation_id'])
        # else:
        #     eval_code = ''
        # code = '\n'.join([code, dedent(eval_code)])

        return code