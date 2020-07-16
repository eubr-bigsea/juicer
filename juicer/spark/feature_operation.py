# -*- coding: utf-8 -*-


import json
from collections import namedtuple
from textwrap import dedent

from juicer.operation import Operation, TransformModelOperation

ScalerNameAndParameters = namedtuple("ScalerNameAndParameters",
                                     "name, parameters, metrics")


class ScalerOperation(Operation):
    """
    Base class for operations to scale data.
    """

    ATTRIBUTE_PARAM = 'attribute'
    ALIAS_PARAM = 'alias'

    __slots__ = ('attribute', 'scaled_attr', 'model')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTE_PARAM in parameters:
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTE_PARAM, self.__class__))
        self.scaled_attr = parameters.get(self.ALIAS_PARAM,
                                          'scaled_{}'.format(self.order))

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.model = self.named_outputs.get('transformation model',
                                            'model_task_{}'.format(self.order))

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.model])

    def get_data_out_names(self, sep=','):
        return self.output

    def _get_scaler_algorithm_and_parameters(self):
        raise NotImplementedError(_('Must be implemented in children classes'))

    def generate_code(self):
        name_and_params = self._get_scaler_algorithm_and_parameters()

        name_and_params.parameters['outputCol'] = self.scaled_attr

        input_data = self.named_inputs['input data']

        code = dedent("""
            params = {params}
            metrics = {metrics}
            scaler = {scaler_impl}(**params)

            features = {features}
            keep = [c.name for c in {input}.schema] + [params['outputCol']]

            # handle categorical features (if it is the case)
            {model} = assemble_features_pipeline_model(
                {input}, features, None, scaler, 'setInputCol', None, None,
                keep, emit_event, '{task_id}')


            # Lazy execution in case of sampling the data in UI
            def call_transform(df):
                return {model}.transform(df)
            {output} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {input}, call_transform).select(keep)
            
            if metrics:
                from juicer.spark.reports import SimpleTableReport
                if isinstance({model}, PipelineModel):
                    scaler_model = {model}.stages[0].stages[2]
                else:
                    scaler_model = {model}

                headers = ['Metric', 'Value']
                rows = [[m, getattr(scaler_model, m)] for m in metrics]

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """.format(
            params=name_and_params.parameters,  # str representation is ok here
            scaler_impl=name_and_params.name,
            features=json.dumps(self.attribute),
            input=input_data,
            output=self.output,
            model=self.model,
            metrics=name_and_params.metrics,
            task_id=self.parameters['task_id'],
            operation_id=self.parameters['operation_id'],
            title=_("Metrics for task"),
        ))

        return code


class StandardScalerOperation(ScalerOperation):
    """
    Standardizes features by removing the mean and scaling to unit variance
    using column summary statistics on the samples in the training set.
    """

    WITH_MEAN_PARAM = 'with_mean'
    WITH_STD_PARAM = 'with_std'
    __slots__ = ('with_std', 'with_mean')

    def __init__(self, parameters, named_inputs, named_outputs):
        ScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.with_mean = parameters.get(
            self.WITH_MEAN_PARAM, False) in ['1', 1, True]
        self.with_std = parameters.get(
            self.WITH_STD_PARAM, True) in ['1', 1, True]

    def _get_scaler_algorithm_and_parameters(self):
        return ScalerNameAndParameters(
            'StandardScaler',
            {'withStd': self.with_std, 'withMean': self.with_mean},
            ['mean', 'std', ]
        )


class MaxAbsScalerOperation(ScalerOperation):
    """
    Rescale each feature individually to range [-1, 1] by dividing through the
    largest maximum absolute value in each feature. It does not shift/center the
    data, and thus does not destroy any sparsity.
    """
    __slots__ = []

    def __init__(self, parameters, named_inputs, named_outputs):
        ScalerOperation.__init__(self, parameters, named_inputs, named_outputs)

    def _get_scaler_algorithm_and_parameters(self):
        return ScalerNameAndParameters('MaxAbsScaler', {}, [])


class MinMaxScalerOperation(ScalerOperation):
    """
    Rescale each feature individually to range [-1, 1] by dividing through the
    largest maximum absolute value in each feature. It does not shift/center the
    data, and thus does not destroy any sparsity.
    """
    __slots__ = ['min', 'max']
    MIN_PARAM = 'min'
    MAX_PARAM = 'max'

    def __init__(self, parameters, named_inputs, named_outputs):
        ScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.min = float(parameters.get(self.MIN_PARAM, 0.0))
        self.max = float(parameters.get(self.MAX_PARAM, 1.0))

    def _get_scaler_algorithm_and_parameters(self):
        return ScalerNameAndParameters(
            'MinMaxScaler', {'min': self.min, 'max': self.max}, [])


class BucketizerOperation(TransformModelOperation):
    """
    From Spark documentation:
    Maps a column of continuous features to a column of feature buckets.
    """
    __slots__ = ['handle_invalid', 'attributes', 'aliases', 'splits', 'model']
    HANDLE_INVALID_PARAM = 'handle_invalid'
    ATTRIBUTES_PARAM = 'attributes'
    ALIASES_PARAM = 'aliases'
    SPLITS_PARAM = 'splits'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.handle_invalid = self.parameters.get(self.HANDLE_INVALID_PARAM)
        if self.handle_invalid is not None:
            if self.handle_invalid not in ['skip', 'keep', 'error']:
                raise ValueError(
                    _('Parameter {} must be one of these: {}').format(
                        _('type'), ','.join([_('keep'), _('skip'), _('error')])
                    )
                )
        self.aliases = self._get_aliases(
            self.attributes, parameters.get(self.ALIASES_PARAM, '').split(','),
            'bucketed')

        self.splits = self._get_splits(parameters)
        self.model = self.named_outputs.get(
            'model', 'model_task_{}'.format(self.order))

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.has_code = [len(self.named_inputs) > 0, self.contains_results()]

    def _get_splits(self, parameters):
        splits = []
        for split in parameters.get(self.SPLITS_PARAM, '').split(','):
            if split == '-INF':
                splits.append(-float("inf"))
            elif split == 'INF':
                splits.append(float("inf"))
            else:
                try:
                    splits.append(float(split))
                except Exception as e:
                    raise ValueError(_('Invalid value for {}: "{}".').format(
                        _('splits'), split))
        if len(splits) < 3:
            raise ValueError(
                _('You must inform at least {} '
                  'values for parameter {}.').format(3, _('splits')))
        if not all(splits[i] < splits[i + 1] for i in
                   range(len(splits) - 1)):
            raise ValueError(
                _('Values for {} must be sorted in ascending order.').format(
                    _('splits')))
        return splits

    def generate_code(self):
        input_data = self.named_inputs['input data']
        splits = []
        for v in self.splits:
            if v == float('inf'):
                splits.append('float("inf")')
            elif v == -float('inf'):
                splits.append('-float("inf")')
            else:
                splits.append(str(v))

        code = dedent("""
            col_alias = dict(tuple({alias}))
            splits = [{splits}]
            bucketizers = [feature.Bucketizer(
                splits=splits, handleInvalid='{handle_invalid}', inputCol=col,
                outputCol=alias, ) for col, alias in col_alias.items()]
            pipeline = Pipeline(stages=bucketizers)

            {model} = pipeline.fit({input})
            {output} = {model}.transform({input})
        """.format(
            alias=json.dumps(list(zip(self.attributes, self.aliases))),
            handle_invalid=self.handle_invalid,
            splits=', '.join(splits),
            input=input_data,
            output=self.output,
            model=self.model,
        ))
        return code


class QuantileDiscretizerOperation(TransformModelOperation):
    """
    From Spark documentation:
    QuantileDiscretizer takes a column with continuous features and outputs a
    column with binned categorical features. The number of bins can be set
    using the numBuckets parameter. It is possible that the number of buckets
    used will be less than this value, for example, if there are too few
    distinct values of the input to create enough distinct quantiles.
    """
    __slots__ = ['relative_error', 'attributes', 'aliases', 'buckets', 'model']
    RELATIVE_ERROR_PARAM = 'relative_error'
    ATTRIBUTES_PARAM = 'attributes'
    ALIASES_PARAM = 'aliases'
    BUCKETS_PARAM = 'buckets'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        try:
            self.relative_error = float(self.parameters.get(
                self.RELATIVE_ERROR_PARAM, '0.001'))
        except:
            raise ValueError(
                _('Invalid value for {}: {}').format(
                    _('relative_error'), self.parameters.get(
                        self.RELATIVE_ERROR_PARAM)))

        self.aliases = self._get_aliases(
            self.attributes, parameters.get(self.ALIASES_PARAM, '').split(','),
            'quantile')

        try:
            self.buckets = int(parameters.get(self.BUCKETS_PARAM, 2))
        except:
            raise ValueError(
                _('Invalid value for {}: {}').format(
                    _('buckets'), self.parameters.get(
                        self.RELATIVE_ERROR_PARAM)))

        self.model = self.named_outputs.get(
            'model', 'model_task_{}'.format(self.order))

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.has_code = [len(self.named_inputs) > 0, self.contains_results()]

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
                col_alias = dict(tuple({alias}))
                qds = [feature.QuantileDiscretizer(
                    numBuckets={buckets}, relativeError={relative_error},
                    inputCol=col, outputCol=alias, )
                    for col, alias in col_alias.items()]
                pipeline = Pipeline(stages=qds)
                {model} = pipeline.fit({input})
                {output} = {model}.transform({input})
            """.format(
            alias=json.dumps(list(zip(self.attributes, self.aliases))),
            relative_error=self.relative_error,
            buckets=repr(self.buckets),
            input=input_data,
            output=self.output,
            model=self.model,
        ))
        return code


class ChiSquaredSelectorOperation(Operation):
    """
    Chi-Squared feature selection, which selects categorical features to use
    for predicting a categorical label
    """

    ATTRIBUTES_PARAM = 'attributes'
    LABEL_PARAM = 'label'
    ALIAS_PARAM = 'alias'
    SELECTOR_TYPE_PARAM = 'selector_type'
    NUM_TOP_FEATURES_PARAM = 'num_top_features'
    PERCENTILE_PARAM = 'percentile'
    FPR_PARAM = 'fpr'
    FDR_PARAM = 'fdr'
    FWE_PARAM = 'fwe'
    VALID_TYPES = ['numTopFeatures', 'percentile', 'fpr', 'fdr', 'fwe']

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        if self.LABEL_PARAM in parameters:
            self.label = parameters.get(self.LABEL_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.LABEL_PARAM, self.__class__))

        self.alias = parameters.get(self.ALIAS_PARAM, 'chi_output')

        if self.SELECTOR_TYPE_PARAM in parameters:
            self.selector_type = parameters.get(self.SELECTOR_TYPE_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.SELECTOR_TYPE_PARAM, self.__class__))
        if self.selector_type not in self.VALID_TYPES:
            raise ValueError(
                _('Parameter {} must be one of these: {}').format(
                    self.SELECTOR_TYPE_PARAM, ','.join(self.VALID_TYPES)
                )
            )

        self.num_top_features = int(
            parameters.get(self.NUM_TOP_FEATURES_PARAM, 50))
        self.percentile = float(parameters.get(self.PERCENTILE_PARAM, 0.1))
        self.fpr = float(parameters.get(self.FPR_PARAM, 0.05))
        self.fdr = float(parameters.get(self.FDR_PARAM, 0.05))
        self.fwe = float(parameters.get(self.FWE_PARAM, 0.05))

        self.model = self.named_outputs.get(
            'model', 'model_task_{}'.format(self.order))

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.has_code = [len(self.named_inputs) > 0, self.contains_results()]

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
                emit = functools.partial(
                    emit_event, name='update task',
                    status='RUNNING', type='TEXT',
                    identifier='{task_id}',
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id},
                    task={{'id': '{task_id}'}},
                    title='{title}')

                features = {features}
                to_assemble = []
                keep_at_end = [c.name for c in {input}.schema]
                keep_at_end.append('{alias}')
                requires_pipeline = False

                stages = []
                if len(features) > 1 and not isinstance(
                    {input}.schema[str(features[0])].dataType, VectorUDT):
                    emit(message='{msg0}')
                    for f in features:
                        if not dataframe_util.is_numeric({input}.schema, f):
                            name = f + '_tmp'
                            to_assemble.append(name)
                            stages.append(feature.StringIndexer(
                                inputCol=f, outputCol=name,
                                handleInvalid='keep'))
                        else:
                            to_assemble.append(f)

                    # Remove rows with null (VectorAssembler doesn't support it)
                    cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                        for c in to_assemble])
                    stages.append(SQLTransformer(
                        statement='SELECT * FROM __THIS__ WHERE {{}}'.format(
                            cond)))

                    final_features = 'features_tmp'
                    stages.append(feature.VectorAssembler(
                        inputCols=to_assemble, outputCol=final_features))
                    requires_pipeline = True
                else:
                    final_features = features[0]

                label_col = '{label}'
                if not dataframe_util.is_numeric({input}.schema, label_col):
                    final_label = label_col + '_tmp'
                    stages.append(feature.StringIndexer(
                                inputCol=label_col, outputCol=final_label,
                                handleInvalid='keep'))
                    requires_pipeline = True
                else:
                    final_label = label_col

                selector = ChiSqSelector(
                    numTopFeatures={top_features}, featuresCol=final_features,
                    outputCol='{alias}', labelCol=final_label,
                    selectorType='{selector_type}', percentile={percentile},
                    fpr={fpr}, fdr={fdr}, fwe={fwe})

                if requires_pipeline:
                    stages.append(selector)
                    # Remove temporary columns
                    sql = 'SELECT {{}} FROM __THIS__'.format(', '.join(
                        keep_at_end))
                    stages.append(SQLTransformer(statement=sql))

                    pipeline = Pipeline(stages=stages)
                    {model} = pipeline.fit({input})
                    chi_model = {model}.stages[-2]
                else:
                    {model} = selector.fit({input})
                    chi_model = {model}

                {output} = {model}.transform({input})

                content = '<h5>{title}</h5>'
                tmp = {input}.select(to_assemble).head(1)[0]
                tmp = [len(i.toArray()) if hasattr(i, 'toArray') 
                        else 1 for i in tmp ]

                import numpy as np
                cumsum = np.cumsum(tmp)
                final_idx = []
                for idx in chi_model.selectedFeatures:
                    final_idx.append(cumsum.searchsorted(idx, side='right'))
    
                content += ', '.join(
                    [{input}.select(to_assemble).schema[int(inx)].name for inx in
                        final_idx])
                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content,
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})

            """.format(
            alias=self.alias,
            label=self.label[0],
            input=input_data,
            output=self.output,
            model=self.model,
            selector_type=self.selector_type,
            percentile=self.percentile,
            top_features=self.num_top_features,
            fpr=self.fpr,
            fdr=self.fdr,
            fwe=self.fwe,
            title=_('Selected features'),
            features=json.dumps(self.attributes),
            task_id=self.parameters['task_id'],
            operation_id=self.parameters['operation_id'],
            msg0=_('Features are not assembled as a vector. They will be '
                   'implicitly assembled and rows with null values will be '
                   'discarded. If this is undesirable, explicitly add a '
                   'feature assembler in the workflow.'),
        ))
        return code
