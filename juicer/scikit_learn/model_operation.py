from textwrap import dedent
from juicer.operation import Operation
import string


class SafeDict(dict):
    # noinspection PyMethodMayBeStatic
    def __missing__(self, key):
        return '{' + key + '}'


class ApplyModelOperation(Operation):
    NEW_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) == 2, self.contains_results()])

        self.new_attribute = parameters.get(self.NEW_ATTRIBUTE_PARAM,
                                            'new_attribute')

        self.feature = parameters['features'][0]
        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

    def get_data_out_names(self, sep=','):
        return self.output

    def generate_code(self):
        input_data1 = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        model = self.named_inputs.get(
            'model', 'model_task_{}'.format(self.order))

        code = dedent("""
            {out} = {in1}
            X_train = {in1}['{features}'].values.tolist()
            {out}['{new_attr}'] = {in2}.predict(X_train).tolist()
            """.format(out=output, in1=input_data1, in2=model,
                       new_attr=self.new_attribute, features=self.feature))

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

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = 'name' in parameters and len(named_inputs) == 1
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('name', self.__class__))
        self.filename = self.parameters['name']
        self.overwrite = parameters.get('write_nome', 'OVERWRITE')
        if self.overwrite == 'OVERWRITE':
            self.overwrite = True
        else:
            self.overwrite = False

    def generate_code(self):
        """Generate code."""
        code = """
        import pickle
        filename = '{filename}'
        pickle.dump({IN}, open(filename, 'wb'))

        """.format(IN=self.named_inputs['input data'],
                   filename=self.filename, overwrite=self.overwrite)
        return dedent(code)


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
        self.feature_attribute = (parameters.get(
                self.FEATURE_ATTRIBUTE_PARAM) or [''])[0]
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

        if self.type_model == 'clustering':
            if self.feature_attribute == '':
                msg = \
                    _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                        self.FEATURE_ATTRIBUTE_PARAM, self.__class__))
            else:
                self.label_attribute = self.feature_attribute
        else:
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

        self.model = self.named_inputs.get('model')
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
                """)]

            if self.type_model == 'classification':
                self._get_code_for_classification_metrics(code)
            elif self.type_model == 'regression':
                self._get_code_for_regression_metrics(code)
            elif self.type_model == 'clustering':
                self._get_code_for_clustering_metrics(code)

            self._get_code_for_summary(code)

            # # Common for all metrics!
            # code.append(dedent("""
            # # {model_output} = ModelsEvaluationResultList(
            # #        [{model}], {model}, '{type_model}', metric_value)
            #
            # #{metric} = metric_value
            # {model_output} = None
            # """))

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
            print (code)
            return dedent(code)

    @staticmethod
    def _get_code_for_classification_metrics(code):
        """
        Generate code for other classification metrics besides those related to
        area.
        """
        code.append(dedent("""
            # classification metrics
            y_pred = {input}[prediction_col].values.tolist()
            y_true = {input}[label_col].values.tolist()
           
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
                    ['Jaccard similarity coefficient score', 
                     jaccard_similarity_score(y_true, y_pred)],
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
            y_pred = {input}[prediction_col].values.tolist()
            y_true = {input}[label_col].values.tolist()

            if display_text:
                headers = {table_headers}
                rows = [
                    ['Silhouette Coefficient',silhouette_score(y_true, y_pred)],
                    ['Calinski and Harabaz score', 
                     calinski_harabaz_score(y_true, y_pred)],
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
            y_pred = {input}[prediction_col].values.tolist()
            y_true = {input}[label_col].values.tolist()
           
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

                identity = range(int(max(y_true[-1], y_pred[-1])))
                emit_event(
                     'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=report2.plot(
                        '{plot_title}',
                        '{plot_x_title}',
                        '{plot_y_title}',
                        identity, identity, 'r.',
                        y_true, y_pred,'b.',),
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
        #code.append(
        #        dedent(string.Formatter().vformat(partial_code, (), i18n_dict)))


class ModelsEvaluationResultList:
    """ Stores a list of ModelEvaluationResult """

    def __init__(self, models, best, metric_name, metric_value):
        self.models = models
        self.best = best
        self.metric_name = metric_name
        self.metric_value = metric_value
