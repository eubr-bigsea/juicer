# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
import re
import pandas as pd
from juicer.util import dataframe_util


class OutlierDetectionOperation(Operation):

    NUMBER_NEIGHBORS_ATTRIBUTE_PARAM = 'n_neighbors'
    ALGORITHM_ATTRIBUTE_PARAM = 'algorithm'
    LEAF_SIZE_ATTRIBUTE_PARAM = 'leaf_size'
    METRIC_ATTRIBUTE_PARAM = 'metric'
    CONTAMINATION_ATTRIBUTE_PARAM = 'contamination'
    P_ATTRIBUTE_PARAM = 'p'
    METRIC_PARAMS_ATTRIBUTE_PARAM = 'metric_params'
    NOVELTY_ATTRIBUTE_PARAM = 'novelty'
    N_JOBS_ATTRIBUTE_PARAM = 'n_jobs'
    FEATURES_PARAM = 'features'
    OUTLIER_PARAM = 'outlier'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.number_neighbors = int(parameters.get(self.NUMBER_NEIGHBORS_ATTRIBUTE_PARAM, 20))
        self.algorithm = parameters.get(self.ALGORITHM_ATTRIBUTE_PARAM, "auto")
        self.leaf_size = int(parameters.get(self.LEAF_SIZE_ATTRIBUTE_PARAM, 30))
        self.metric = parameters.get(self.METRIC_ATTRIBUTE_PARAM, "minkowski")
        self.contamination = float(parameters.get(self.CONTAMINATION_ATTRIBUTE_PARAM, 0.22))
        self.p = int(parameters.get(self.P_ATTRIBUTE_PARAM, 2))
        self.metric_params = parameters.get(self.METRIC_PARAMS_ATTRIBUTE_PARAM, None)
        self.novelty = int(parameters.get(self.NOVELTY_ATTRIBUTE_PARAM, 0))
        self.n_jobs = int(parameters.get(self.N_JOBS_ATTRIBUTE_PARAM, 0))
        self.features = parameters['features']
        self.outlier = self.parameters.get(self.OUTLIER_PARAM, 'outlier')

        self.input_treatment()
        self.transpiler_utils.add_import(
            "from sklearn.neighbors import LocalOutlierFactor")

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def input_treatment(self):
        if self.number_neighbors <= 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.NUMBER_NEIGHBORS_ATTRIBUTE_PARAM, self.__class__))
        if self.leaf_size <= 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.LEAF_SIZE_ATTRIBUTE_PARAM, self.__class__))
        if self.contamination < 0 or self.contamination > 0.5:
            raise ValueError(
                _("Parameter '{}' must be x>=0 and x<=0.5 for task {}").format(
                    self.CONTAMINATION_ATTRIBUTE_PARAM, self.__class__))
        if self.metric is 'minkowski':
            if self.p <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x>0 for task {}").format(
                        self.P_ATTRIBUTE_PARAM, self.__class__))

        self.novelty = True if int(self.novelty) == 1 else False

        self.n_jobs = 1 if int(self.n_jobs) == 0 else int(self.n_jobs)

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
            {output_data} = {input}{copy_code}
            if '{metric}' is 'minkowski':
                clf = LocalOutlierFactor(n_neighbors={number_neighbors}, contamination={contamination}, 
                                             metric='{metric}', algorithm='{algorithm}', n_jobs={n_jobs}, 
                                             leaf_size={leaf_size}, novelty={novelty}, p={p},
                                             metric_params={metric_params})
            else:
                clf = LocalOutlierFactor(n_neighbors={number_neighbors}, contamination={contamination}, 
                                             metric='{metric}', algorithm='{algorithm}', n_jobs={n_jobs}, 
                                             leaf_size={leaf_size}, novelty={novelty}, metric_params={metric_params})
            
            X = {input}[{columns}].values.tolist()
            p = clf.fit_predict(X).astype(float)
            clf2 = clf.negative_outlier_factor_ 
            T2 = pd.DataFrame(p, columns = ['{outlier}'])
            {output_data} = pd.concat([{input},T2],axis=1)
            """.format(copy_code=copy_code, output=self.output,
                       input=self.named_inputs['input data'],
                       number_neighbors=self.number_neighbors,
                       contamination=self.contamination,
                       metric=self.metric,
                       algorithm=self.algorithm,
                       n_jobs=self.n_jobs,
                       leaf_size=self.leaf_size,
                       novelty=self.novelty,
                       p=self.p,
                       metric_params=self.metric_params,
                       output_data=self.output,
                       columns=self.features,
                       outlier=self.outlier)

        return dedent(code)