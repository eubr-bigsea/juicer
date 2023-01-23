from itertools import zip_longest as zip_longest
from textwrap import dedent

import juicer.scikit_learn.feature_operation as sk_feature


class FeatureAssemblerOperation(sk_feature.FeatureAssemblerOperation):
    """Assemble features as a vector/list
    Since 2.6
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        sk_feature.FeatureAssemblerOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        self.input = self.parameters[self.ATTRIBUTES_PARAM]
        code = f"""
        features = {self.attributes}
        numeric_types = {{
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
            pl.Float32, pl.Float64,}}
        test_invalid = lambda t: (t not in numeric_types and 
            t != pl.List t.inner not in numeric_types)

        invalid = [c for i, c in enumerate({self.input}.columns) 
            if c in features and test_invalid({self.input}.dtypes[i])]
        if invalid:
            raise ValueError("Input '{self.input}' must contain only numeric "
            " values for task {self.__class__}")
        # Nulls and NaN are assembled. FIXME: evaluate a property to skip 
        # or keep such values.
        {self.output} = {self.input}.select(
            pl.concat_list(features).alias('{self.alias}'))
        """
        return dedent(code)


class FeatureDisassemblerOperation(sk_feature.FeatureDisassemblerOperation):
    """ 
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.FeatureDisassemblerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        code = f"""
        {self.output} = {self.named_inputs['input data']}.select(
            pl.exclude('{self.feature}'),
            pl.col('{self.feature}').arr
                .to_struct(n_field_strategy="max_width")).unnest(
                    '{self.feature}')
        """
        return dedent(code)


class MinMaxScalerOperation(sk_feature.MinMaxScalerOperation):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e.
    between zero and one.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    Since 2.6
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        sk_feature.MinMaxScalerOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        input1 = self.named_inputs['input data']
        code = f"""
            attributes = {self.attributes}
            # Second select: restores attributes original order
            {self.output} = {input1}.select(
                [pl.exclude(attributes)] + 
                [((pl.col(c) - pl.col(c).min()) /
                        (pl.col(c).max() - pl.col(c).min()) 
                            * ({self.max} - {self.min}) + {self.min}).alias(c)
                for c in attributes]).select({input1}.columns)
        """

        return dedent(code)


class MaxAbsScalerOperation(sk_feature.MaxAbsScalerOperation):
    """
    Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the training
    set will be 1.0. It does not shift/center the data, and thus does not
    destroy any sparsity.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.MaxAbsScalerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        """Generate code."""
        if not self.has_code:
            return None

        input1 = self.named_inputs['input data']
        code = f"""
            attributes = {self.attributes}
            # Second select: restores attributes original order
            {self.output} = {input1}.select(
                [pl.exclude(attributes)] + 
                [(pl.col(c) / pl.col(c).abs().max()).alias(c)
                    for c in attributes]).select({input1}.columns)
        """
        return dedent(code)


class StandardScalerOperation(sk_feature.StandardScalerOperation):
    """
    Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Standardization of a dataset is a common requirement for many machine
    learning estimators: they might behave badly if the individual feature
    do not more or less look like standard normally distributed data.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.StandardScalerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        input1 = self.named_inputs['input data']
        return dedent(f"""
            attributes = {self.attributes}
            # Second select: restores attributes original order
            {self.output} = {input1}.select(
                [pl.exclude(attributes)] + 
                [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
                    for c in attributes]).select({input1}.columns)
        """)


class KBinsDiscretizerOperation(sk_feature.KBinsDiscretizerOperation):
    """
    Transform features using Kbins discretizer.
    Discretizes features into k bins:
    """

    template = """
        {%- if strategy == 'uniform' %}
        min_value = math.floor({{input}}.get_column('{{attr}}').min())
        max_value = math.floor({{input}}.get_column('{{attr}}').max())
        breaks = np.linspace(min_value, max_value, {{n_quantiles}}).tolist()
        {{output}} = {{input}}.insert_at_idx(
            len({{input}}.columns),
            pl.cut(
                {{input}}, 
                breaks, 
                category_label='{{alias}}').select(
                    pl.col('{{attr}}').rank('dense')).get_column('{{attr}}'))
        {%- else %}
        {{output}} = {{input}}.with_column(
            {{input}}.select(((pl.col('{{attr}}').rank('dense') - 1) 
                % {{n_quantiles}})).alias('{{alias}}'))
        {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.KBinsDiscretizerOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        if self.output_distribution:
            self.transpiler_utils.add_import("import math")
            self.transpiler_utils.add_import("import numpy as np")

    def generate_code(self):
        """Generate code."""
        if not self.has_code:
            return None
        ctx = dict(output=self.output,
                   model=self.model,
                   input=self.named_inputs['input data'],
                   strategy=self.output_distribution,
                   att=self.attribute, alias=self.alias,
                   n_quantiles=self.n_quantiles,)

        return dedent(self.render_template(ctx))


class OneHotEncoderOperation(sk_feature.OneHotEncoderOperation):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    This encoding is needed for feeding categorical data to many
    scikit-learn estimators, notably linear models and SVMs with
    the standard kernels.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        """Generate code."""
        if not self.has_code:
            return None

        self.input = self.named_inputs['input data']
        return dedent(f"""
            {self.output} = {self.input}.with_column(
                ({self.input}.select(pl.col('{self.attribute}'))
                    .to_dummies()
                    .select(pl.concat_list(pl.all()))
                    .to_series()
                    .alias('{self.alias}'))
        """)


class PCAOperation(sk_feature.PCAOperation):
    """
    Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value
    Decomposition of the data to project it to a lower dimensional space.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    N_COMPONENTS = 'k'

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.PCAOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = """ 
            {output} = {input}{copy_code}
            pca = PCA(n_components={n_comp})
            X_train = get_X_train_data({input}, {att})
            {output}['{alias}'] = pca.fit_transform(X_train).tolist()
            """.format(copy_code=copy_code, output=self.output,
                       input=self.named_inputs['input data'],
                       att=self.attributes, alias=self.alias,
                       n_comp=self.n_components)
            return dedent(code)


class LSHOperation(sk_feature.LSHOperation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        sk_feature.LSHOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

    def generate_code(self):
        input_data = self.named_inputs['input data']
        """Generate code."""
        # TODO: LSHForest algorithm is using all columns.

        # TODO: Is this working?

        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
            {output} = {input}{copy_code}
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
                       model=self.model)

        return dedent(code)


class StringIndexerOperation(sk_feature.StringIndexerOperation):
    """
    A label indexer that maps a string attribute of labels to an ML attribute of
    label indices (attribute type = STRING) or a feature transformer that merges
    multiple attributes into a vector attribute (attribute type = VECTOR). All
    other attribute types are first converted to STRING and them indexed.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.StringIndexerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        input_data = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        # FIXME: Models are not supported
        models = self.named_outputs.get('models',
                                        'models_task_{}'.format(self.order))
        code = f"""
            # Models are not supported
            {models} = dict()
            attr_alias = zip({self.attributes}, {self.alias})
            {output} = {input_data}.select([pl.all()] +  
                [ pl.col(attr).over(attr).rank('dense').alias(alias)
                    for attr, alias in attr_alias])
            """
        return dedent(code)
