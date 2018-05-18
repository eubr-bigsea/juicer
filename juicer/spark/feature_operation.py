# -*- coding: utf-8 -*-
from __future__ import unicode_literals

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
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM)[0]
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
        name_and_params.parameters['inputCol'] = self.attribute
        name_and_params.parameters['outputCol'] = self.scaled_attr

        input_data = self.named_inputs['input data']

        code = dedent("""
            params = {params}
            metrics = {metrics}
            scaler = {scaler_impl}(**params)
            {model} = scaler.fit({input})

            # Lazy execution in case of sampling the data in UI
            def call_transform(df):
                return {model}.transform(df)
            {output} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {input}, call_transform)
            if metrics:
                from juicer.spark.reports import SimpleTableReport
                headers = ['Metric', 'Value']
                rows = [[m, getattr({model}, m)] for m in metrics]

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
                   xrange(len(splits) - 1)):
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
            alias=json.dumps(zip(self.attributes, self.aliases)),
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
            alias=json.dumps(zip(self.attributes, self.aliases)),
            relative_error=self.relative_error,
            buckets=repr(self.buckets),
            input=input_data,
            output=self.output,
            model=self.model,
        ))
        return code
