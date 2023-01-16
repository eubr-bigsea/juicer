from itertools import zip_longest as zip_longest
from textwrap import dedent

from juicer.operation import Operation
from juicer.scikit_learn.util import get_X_train_data

import juicer.scikit_learn.feature_operation as sk_feature


class FeatureAssemblerOperation(sk_feature.FeatureAssemblerOperation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        sk_feature.FeatureAssemblerOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

    def generate_code(self):
        if self.has_code:
            code = """
            cols = {cols}
            if {input}[cols].dtypes.all() == np.object:
                raise ValueError("Input '{input}' must contain numeric values"
                " only for task {cls}")

            {output} = {input}.dropna(subset=cols)
            {output}['{alias}'] = {output}[cols].to_numpy().tolist()
            """.format(output=self.output, alias=self.alias,
                       input=self.named_inputs['input data'],
                       cols=self.parameters[self.ATTRIBUTES_PARAM],
                       cls=self.__class__)
            return dedent(code)


class FeatureDisassemblerOperation(sk_feature.FeatureDisassemblerOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.FeatureDisassemblerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        if self.has_code:

            code = """
            {input} = {input}.reset_index(drop=True)
            feature = {input}['{feature}'].to_numpy()
            tmp_vec = np.stack(feature, axis=0)
            dim = tmp_vec.shape[1] if {topn} > tmp_vec.shape[1] else {topn}
            columns = ["{alias}"+str(i+1) for i in range(dim)]
            new_df = pd.DataFrame(tmp_vec[:,:dim], columns=columns)
            {output} = {input}.merge(new_df, left_index=True, right_index=True)
            """.format(output=self.output,
                       alias=self.alias,
                       topn=self.topn,
                       input=self.named_inputs['input data'],
                       feature=self.feature)
            return dedent(code)


class MinMaxScalerOperation(sk_feature.MinMaxScalerOperation):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e.
    between zero and one.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        sk_feature.MinMaxScalerOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        input1 = self.named_inputs['input data']
        code = f"""
            X_train = {input1}.select([
                {self.attributes}
            ]).to_numpy()

            {self.model} = MinMaxScaler(feature_range=({self.min}, {self.max}))
            values = pl.Dataframe(
                {self.model}.fit_transform(X_train), 
                columns={self.alias})

            {self.output} = pl.concat(
                [{input1}, values], how='horizontal').lazy()
        """

        return dedent(code)


class MaxAbsScalerOperation(sk_feature.MaxAbsScalerOperation):
    """
    Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the training
    set will be 1.0. It does not shift/center the data, and thus does not
     destroy any sparsity.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.MaxAbsScalerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            code = """
            X_train = get_X_train_data({input}, {att})

            {model} = MaxAbsScaler()
            values = {model}.fit_transform(X_train)

            {output} = pd.concat([{input},
                pd.DataFrame(values, columns={alias})],
                ignore_index=False, axis=1)
            """.format(output=self.output, model=self.model,
                       input=self.named_inputs['input data'],
                       att=self.attributes, alias=self.alias)

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
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.StandardScalerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        if self.has_code:
            """Generate code."""
            op = "with_mean={value}" \
                .format(value=self.with_mean)
            op += ", with_std={value}" \
                .format(value=self.with_std)

            code = """
            X_train = get_X_train_data({input}, {att})

            {model} = StandardScaler({op})
            values = {model}.fit_transform(X_train)

            {output} = pd.concat([{input},
                pd.DataFrame(values, columns={alias})],
                ignore_index=False, axis=1)
            """.format(model=self.model, output=self.output,
                       input=self.named_inputs['input data'],
                       att=self.attributes, alias=self.alias, op=op)

            return dedent(code)


class KBinsDiscretizerOperation(sk_feature.KBinsDiscretizerOperation):
    """
    Transform features using Kbins discretizer.

    This method transforms the features to follow a uniform or a
    normal distribution. Therefore, for a given feature, this transformation
    tends to spread out the most frequent values.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.KBinsDiscretizerOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        if self.has_code:
            """Generate code."""
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = """
            {output} = {input}{copy_code}
            {model} = KBinsDiscretizer(n_bins={n_quantiles}, 
                encode='ordinal', strategy='{strategy}')
            X_train = get_X_train_data({input}, {att})
            
            values = {model}.fit_transform(X_train)

            {output} = pd.concat([{input}, 
                pd.DataFrame(values, columns={alias})],
                ignore_index=False, axis=1)
            """.format(copy_code=copy_code, output=self.output,
                       model=self.model,
                       input=self.named_inputs['input data'],
                       strategy=self.output_distribution,
                       att=self.attribute, alias=self.alias,
                       n_quantiles=self.n_quantiles,)
            return dedent(code)


class OneHotEncoderOperation(sk_feature.OneHotEncoderOperation):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    This encoding is needed for feeding categorical data to many
    scikit-learn estimators, notably linear models and SVMs with
    the standard kernels.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        sk_feature.OneHotEncoderOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = """
            {output} = {input}{copy_code}
            enc = OneHotEncoder()
            X_train = get_X_train_data({input}, {att})
            {output}['{alias}'] = enc.fit_transform(X_train).toarray().tolist()
            """.format(copy_code=copy_code, output=self.output,
                       input=self.named_inputs['input data'],
                       att=self.attribute, alias=self.alias)
            return dedent(code)


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
        if self.has_code:
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
