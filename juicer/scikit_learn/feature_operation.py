from textwrap import dedent
from juicer.operation import Operation
import json
try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import zip_longest as zip_longest


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
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""
        code = """
        cols = {cols}
        {input}_without_na = {input}.dropna(subset=cols)
        {output} = {input}_without_na{copy_code}
        {output}['{alias}'] = {input}_without_na[cols].to_numpy().tolist()
        """.format(copy_code=copy_code, output=self.output, alias=self.alias,
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
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """        
        {model} = MinMaxScaler(feature_range=({min},{max}))
        X_train = get_X_train_data({input}, {att})
        {model}.fit(X_train)
        """.format(output=self.output, model=self.model,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias,
                   min=self.min, max=self.max)

        # TODO: corrigir essa checagem
        if self.contains_results() or len(self.named_outputs) > 0:
            code += """
            {output} = {input}{copy_code}
            {output}['{alias}'] = {model}.transform(X_train).tolist()
            """.format(copy_code=copy_code, output=self.output,
                       input=self.named_inputs['input data'],
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

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.ATTRIBUTE_PARAM not in parameters:
            raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTE_PARAM, self.__class__))

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))
        self.model = named_outputs.get(
            'transformation model', 'model_{}'.format(self.order))
        self.attribute = parameters[self.ATTRIBUTE_PARAM]
        self.alias = parameters.get(self.ALIAS_PARAM,
                                        'scaled_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = """
    from sklearn.preprocessing import MaxAbsScaler
    from juicer.scikit_learn.util import get_X_train_data
    {model} = MaxAbsScaler()
    X_train = get_X_train_data({input}, {att})
    {model}.fit(X_train)
            """.format(output=self.output,
                       model=self.model,
                       input=self.named_inputs['input data'],
                       att=self.attribute, alias=self.alias)

            code += """
    {output} = {input}{copy_code}
    {output}['{alias}'] = {model}.transform(X_train).tolist()
                """.format(copy_code=copy_code,
                           output=self.output, input=self.named_inputs['input data'],
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
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        """Generate code."""
        code = """
        {model} = StandardScaler({op})
        X_train = get_X_train_data({input}, {att})
        {model}.fit(X_train)
        """.format(model=self.model,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias, op=op)

        if self.contains_results() or len(self.named_outputs) > 0:
            code += """
            {output} = {input}{copy_code}
            {output}['{alias}'] = {model}.transform(X_train).tolist()
            """.format(copy_code=copy_code, output=self.output,
                       input=self.named_inputs['input data'],
                       model=self.model, alias=self.alias)
        return dedent(code)


class QuantileDiscretizerOperation(Operation):
    """
    Transform features using quantiles information.

    This method transforms the features to follow a uniform or a
    normal distribution. Therefore, for a given feature, this transformation
    tends to spread out the most frequent values.
    """

    # TODO: in tahiti, change the translation "quantil number" to
    # 'The number of bins to produce'

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    N_QUANTILES_PARAM = 'n_quantiles'
    DISTRIBUITION_PARAM = 'output_distribution'

    DISTRIBUITION_PARAM_KMEANS = 'kmeans'
    DISTRIBUITION_PARAM_UNIFORM = 'uniform'
    DISTRIBUITION_PARAM_QUANTIS = 'quantile'

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
            self.model = self.named_outputs.get(
                    'model', 'model_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        'quantiledisc_{}'.format(self.order))
            self.n_quantiles = parameters.get(
                    self.N_QUANTILES_PARAM, 1000) or 1000
            self.output_distribution = parameters.get(
                    self.DISTRIBUITION_PARAM, self.DISTRIBUITION_PARAM_QUANTIS)\
                or self.DISTRIBUITION_PARAM_QUANTIS

            if int(self.n_quantiles) <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_QUANTILES_PARAM, self.__class__))

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
        {output} = {input}{copy_code}
        from sklearn.preprocessing import KBinsDiscretizer
        {model} = KBinsDiscretizer(n_bins={n_quantiles}, 
            encode='ordinal', strategy='{strategy}')
        X_train = get_X_train_data({input}, {att})
        
        {output}['{alias}'] = {model}.fit_transform(X_train).flatten().tolist()
        """.format(copy_code=copy_code, output=self.output,
                   model=self.model,
                   input=self.named_inputs['input data'],
                   strategy=self.output_distribution,
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
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
        {output} = {input}{copy_code}
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        X_train = get_X_train_data({input}, {att})
        {output}['{alias}'] = enc.fit_transform(X_train).toarray().tolist()
        """.format(copy_code=copy_code, output=self.output,
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
            self.n_components = int(parameters[self.N_COMPONENTS])

            self.output = self.named_outputs.get(
                    'output data',
                    'output_data_{}'.format(self.order))
            self.alias = parameters.get(self.ALIAS_PARAM, 'pca_feature')
            if self.n_components <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_COMPONENTS, self.__class__))

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
        {output} = {input}{copy_code}
        from sklearn.decomposition import PCA
        pca = PCA(n_components={n_comp})
        X_train = get_X_train_data({input}, {att})
        {output}['{alias}'] = pca.fit_transform(X_train).tolist()
        """.format(copy_code=copy_code, output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes, alias=self.alias,
                   n_comp=self.n_components)
        return dedent(code)


class LSHOperation(Operation):

    N_ESTIMATORS_ATTRIBUTE_PARAM = 'n_estimators'
    MIN_HASH_MATCH_ATTRIBUTE_PARAM = 'min_hash_match'
    N_CANDIDATES = 'n_candidates'
    NUMBER_NEIGHBORS_ATTRIBUTE_PARAM = 'n_neighbors'
    RANDOM_STATE_ATTRIBUTE_PARAM = 'random_state'
    RADIUS_ATTRIBUTE_PARAM = 'radius'
    RADIUS_CUTOFF_RATIO_ATTRIBUTE_PARAM = 'radius_cutoff_ratio'
    LABEL_PARAM = 'label'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True

        self.number_neighbors = int(parameters.get(
                self.NUMBER_NEIGHBORS_ATTRIBUTE_PARAM, 5))
        self.n_estimators = int(parameters.get(
                self.N_ESTIMATORS_ATTRIBUTE_PARAM, 10))
        self.min_hash_match = int(parameters.get(
                self.MIN_HASH_MATCH_ATTRIBUTE_PARAM, 4))
        self.n_candidates = int(parameters.get(self.N_CANDIDATES, 10))
        self.random_state = int(parameters.get(
                self.RANDOM_STATE_ATTRIBUTE_PARAM, 0))
        self.radius = float(parameters.get(self.RADIUS_ATTRIBUTE_PARAM, 1.0))
        self.radius_cutoff_ratio = float(parameters.get(
                self.RADIUS_CUTOFF_RATIO_ATTRIBUTE_PARAM, 0.9))

        if not all([self.LABEL_PARAM in parameters]):
            msg = _("Parameters '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.LABEL_PARAM,
                self.__class__.__name__))

        self.label = parameters[self.LABEL_PARAM]
        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))
        self.output = self.named_outputs.get(
            'output data', 'out_task_{}'.format(self.order))

        self.input_treatment()

        self.has_import = \
            """
            from sklearn.neighbors import LSHForest
            """

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.radius_cutoff_ratio < 0 or self.radius_cutoff_ratio > 1:
            raise ValueError(
                _("Parameter '{}' must be x>=0 and x<=1 for task {}").format(
                    self.RADIUS_CUTOFF_RATIO_ATTRIBUTE_PARAM, self.__class__))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        """Generate code."""
        #TODO: LSHForest algorithm is using all columns.

        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
            {output_data} = {input}{copy_code}
            X_train = {input}.to_numpy().tolist()
            lshf = LSHForest(min_hash_match={min_hash_match}, 
                n_candidates={n_candidates}, n_estimators={n_estimators},
                number_neighbors={number_neighbors}, radius={radius}, 
                radius_cutoff_ratio={radius_cutoff_ratio}, 
                random_state={random_state})
            {model} = lshf.fit(X_train) 
            """.format(copy_code=copy_code, output=self.output,
                       input=input_data,
                       number_neighbors=self.number_neighbors,
                       n_estimators=self.n_estimators,
                       min_hash_match=self.min_hash_match,
                       n_candidates=self.n_candidates,
                       random_state=self.random_state,
                       radius=self.radius,
                       radius_cutoff_ratio=self.radius_cutoff_ratio,
                       output_data=self.output,
                       model=self.model)

        return dedent(code)


class StringIndexerOperation(Operation):
    """
    A label indexer that maps a string attribute of labels to an ML attribute of
    label indices (attribute type = STRING) or a feature transformer that merges
    multiple attributes into a vector attribute (attribute type = VECTOR). All
    other attribute types are first converted to STRING and them indexed.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name suffixed by _indexed.
        self.alias = [x[1] or '{}_indexed'.format(x[0]) for x in
                      zip_longest(self.attributes,
                                  self.alias[:len(self.attributes)])]
        self.has_import = \
            """
            from sklearn.preprocessing import LabelEncoder
            """

    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        models = self.named_outputs.get('models',
                                        'models_task_{}'.format(self.order))
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = dedent("""
           {output} = {input}{copy_code}
           {models} = dict()
           le = LabelEncoder()
           for col, new_col in zip({columns}, {alias}):
                data = {input}[col].to_numpy().tolist()
                {models}[new_col] = le.fit_transform(data)
                {output}[new_col] =le.fit_transform(data)    
           """.format(copy_code=copy_code, input=input_data,
                      output=output,
                      models=models,
                      columns=self.attributes,
                      alias=self.alias))
        return code
