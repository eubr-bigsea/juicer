from textwrap import dedent

from juicer.operation import Operation


# TODO: https://spark.apache.org/docs/2.2.0/ml-features.html#vectorassembler
class FeatureAssemblerOperation(Operation):

    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            if self.ATTRIBUTES_PARAM not in parameters:
                raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTES_PARAM, self.__class__))

            self.alias = parameters.get(self.ALIAS_PARAM, 'FeatureField')
            self.output = self.named_outputs.get('output data',
                                                 'output_data_{}'.format(
                                                         self.order))

    def generate_code(self):
        code = """
        cols = {cols}
        {input}_without_na = {input}.dropna(subset=cols)
        {output} = {input}_without_na.copy()
        {output}['{alias}'] = {input}_without_na[cols].to_numpy().tolist()
        """.format(output=self.output, alias=self.alias,
                   input=self.named_inputs['input data'],
                   cols=self.parameters[self.ATTRIBUTES_PARAM])

        return dedent(code)


class MinMaxScalerOperation(Operation):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e.
    between zero and one.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    MIN_PARAM = 'min'
    MAX_PARAM = 'max'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTE_PARAM, self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.model = named_outputs.get(
                'transformation model', 'model_{}'.format(self.order))

            if self.ATTRIBUTE_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.ATTRIBUTE_PARAM, self.__class__.__name__))
            self.attribute = parameters[self.ATTRIBUTE_PARAM]

            self.alias = parameters.get(self.ALIAS_PARAM,
                                        'scaled_{}'.format(self.order))

            self.min = parameters.get(self.MIN_PARAM, 0) or 0
            self.max = parameters.get(self.MAX_PARAM, 1) or 1

            self.has_import = \
                "from sklearn.preprocessing import MinMaxScaler\n"

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """        
        {model} = MinMaxScaler(feature_range=({min},{max}))
        X_train = {input}[{att}].to_numpy().tolist()
        {model}.fit(X_train)
        """.format(output=self.output, model=self.model,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias,
                   min=self.min, max=self.max)

        if self.contains_results() or 'output data' or 'output_data_{}' in self.named_outputs:
            code += """
            {output} = {input}
            {output}['{alias}'] = {model}.transform(X_train).tolist()
            """.format(output=self.output, input=self.named_inputs['input data'],
                       model=self.model, alias=self.alias)
        return dedent(code)


class MaxAbsScalerOperation(Operation):
    """
    Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the training
    set will be 1.0. It does not shift/center the data, and thus does not
     destroy any sparsity.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format(self.ATTRIBUTE_PARAM, self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.model = named_outputs.get(
                'transformation model', 'model_{}'.format(self.order))

            if self.ATTRIBUTE_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.ATTRIBUTE_PARAM, self.__class__.__name__))
            self.attribute = parameters[self.ATTRIBUTE_PARAM]

            self.alias = parameters.get(self.ALIAS_PARAM,
                                        'scaled_{}'.format(self.order))
            self.has_import = \
                "from sklearn.preprocessing import MaxAbsScaler\n"

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
        {model} = MaxAbsScaler()
        X_train = {input}[{att}].to_numpy().tolist()
        {model}.fit(X_train)
        """.format(output=self.output,
                   model=self.model,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias)

        if self.contains_results() or 'output data' or 'output_data_{}' in self.named_outputs:
            code += """
            {output} = {input}
            {output}['{alias}'] = {model}.transform(X_train).tolist()
            """.format(output=self.output, input=self.named_inputs['input data'],
                       model=self.model, alias=self.alias)
        return dedent(code)


class StandardScalerOperation(Operation):
    """
    Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Standardization of a dataset is a common requirement for many machine
    learning estimators: they might behave badly if the individual feature
    do not more or less look like standard normally distributed data.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    WITH_MEAN_PARAM = 'with_mean'
    WITH_STD_PARAM = 'with_std'
    VALUE_PARAMETER = 'value'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) > 0 and any(
            [len(self.named_outputs) > 0, self.contains_results()])
        if self.has_code:
            self.with_mean = parameters.get(
                self.WITH_MEAN_PARAM, False) in ['1', 1, True]
            self.with_std = parameters.get(
                self.WITH_STD_PARAM, True) in ['1', 1, True]
            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.model = named_outputs.get(
                'transformation model', 'model_{}'.format(self.order))

            if self.ATTRIBUTE_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.ATTRIBUTE_PARAM, self.__class__.__name__))
            self.attribute = parameters[self.ATTRIBUTE_PARAM]

            self.alias = parameters.get(self.ALIAS_PARAM,
                                        'scaled_{}'.format(self.order))

            self.has_import = \
                "from sklearn.preprocessing import StandardScaler\n"

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        op = "with_mean={value}" \
            .format(value=self.with_mean)
        op += ", with_std={value}" \
            .format(value=self.with_std)

        """Generate code."""
        code = """
        {model} = StandardScaler({op})
        X_train = {input}[{att}].to_numpy().tolist()
        {model}.fit(X_train)
        """.format(model=self.model,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias, op=op)

        if self.contains_results() or 'output data' or 'output_data_{}' in self.named_outputs:
            code += """
            {output} = {input}
            {output}['{alias}'] = {model}.transform(X_train).tolist()
            """.format(output=self.output, input=self.named_inputs['input data'],
                       model=self.model, alias=self.alias)
        return dedent(code)


class QuantileDiscretizerOperation(Operation):
    """
    Transform features using quantiles information.

    This method transforms the features to follow a uniform or a
    normal distribution. Therefore, for a given feature, this transformation
    tends to spread out the most frequent values.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    N_QUANTILES_PARAM = 'n_quantiles'
    DISTRIBUITION_PARAM = 'output_distribution'
    SEED_PARAM = 'seed'

    DISTRIBUITION_PARAM_NORMAL = 'normal'
    DISTRIBUITION_PARAM_UNIFORM = 'uniform'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format('attributes', self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.model = self.named_outputs.get('model', 'model_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        'quantiledisc_{}'.format(self.order))
            self.n_quantiles = parameters.get(
                    self.N_QUANTILES_PARAM, 1000) or 1000
            self.output_distribution = parameters.get(
                    self.DISTRIBUITION_PARAM, self.DISTRIBUITION_PARAM_UNIFORM)\
                or self.DISTRIBUITION_PARAM_UNIFORM
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            if int(self.n_quantiles) <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_QUANTILES_PARAM, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}.copy()
        from sklearn.preprocessing import KBinsDiscretizer
        {model} = KBinsDiscretizer(n_bins={n_quantiles}, encode='ordinal', strategy='quantile')
        X_train = {input}[{att}].to_numpy().tolist()
        
        {output}['{alias}'] = {model}.fit_transform(X_train).flatten().tolist()
        """.format(output=self.output,
                   model=self.model,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias,
                   n_quantiles=self.n_quantiles,)
        return dedent(code)


class OneHotEncoderOperation(Operation):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    This encoding is needed for feeding categorical data to many
    scikit-learn estimators, notably linear models and SVMs with
    the standard kernels.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format('attributes', self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        'onehotenc_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        X_train = {input}[{att}].to_numpy().tolist()
        {output}['{alias}'] = enc.fit_transform(X_train).toarray().tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias)
        return dedent(code)


class PCAOperation(Operation):
    """
    Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value
    Decomposition of the data to project it to a lower dimensional space.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    N_COMPONENTS = 'k'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format(self.ATTRIBUTE_PARAM, self.__class__))
            self.attributes = parameters[self.ATTRIBUTE_PARAM]
            self.n_components = parameters[self.N_COMPONENTS]

            self.output = self.named_outputs.get(
                    'output data',
                    'output_data_{}'.format(self.order))
            self.alias = parameters.get(self.ALIAS_PARAM, 'pca_feature')
            if int(self.n_components) <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_COMPONENTS, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.decomposition import PCA
        pca = PCA(n_components={n_comp})
        X_train = {input}[{att}].to_numpy().tolist()
        {output}['{alias}'] = pca.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes, alias=self.alias,
                   n_comp=self.n_components)
        return dedent(code)
