# coding=utf-8


import logging
from textwrap import dedent

from juicer import auditing
from juicer.spark.ml_operation import SvmClassifierOperation, \
    LogisticRegressionClassifierOperation, DecisionTreeClassifierOperation, \
    GBTClassifierOperation, NaiveBayesClassifierOperation, \
    RandomForestClassifierOperation, PerceptronClassifier, \
    ClassificationModelOperation, ClusteringModelOperation, \
    KMeansClusteringOperation, KModesClusteringOperation, \
    GaussianMixtureClusteringOperation, \
    LdaClusteringOperation, RegressionModelOperation, LinearRegressionOperation, \
    GeneralizedLinearRegressionOperation, DecisionTreeRegressionOperation, \
    GBTRegressorOperation, AFTSurvivalRegressionOperation, \
    RandomForestRegressorOperation, IsotonicRegressionOperation

try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import zip_longest as zip_longest

from juicer.operation import Operation

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class AlgorithmOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs,
                 model, algorithm):
        super(AlgorithmOperation, self).__init__(
            parameters, named_inputs, named_outputs)
        self.algorithm = algorithm
        self.model = model
        self.apply_one_versus_rest = parameters.get(
            'one_vs_rest') in [1, '1', 'True', True]

        self.has_code = len(self.named_inputs) and any(
            [len(self.named_outputs) > 0, self.contains_results()])

    def generate_code(self):
        algorithm_code = self.algorithm.generate_code() or ''
        model_code = self.model.generate_code() or ''
        if self.apply_one_versus_rest:
            # Only valid for classification
            # Code is generated here because it is simple
            one_vs_rest_code = dedent("""
                algorithm = [classification.OneVsRest(
                    classifier={alg}[0]), {alg}[1], {alg}[2]]
                """.format(alg="algorithm"))
            return "\n".join([algorithm_code, one_vs_rest_code, model_code])
        else:
            return "\n".join([algorithm_code, model_code])

    def get_output_names(self, sep=','):
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))
        models = self.named_outputs.get('model',
                                        'model_task_{}'.format(self.order))
        return sep.join([output, models])

    def get_audit_events(self):
        parent_events = super(AlgorithmOperation, self).get_audit_events()
        return parent_events + [auditing.CREATE_MODEL, auditing.APPLY_MODEL]


class ClassificationOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        model_in_ports = {
            'train input data': named_inputs.get('train input data'),
            'algorithm': 'algorithm'}

        model = ClassificationModelOperation(
            parameters, model_in_ports, named_outputs)
        model.clone_algorithm = False
        super(ClassificationOperation, self).__init__(
            parameters, named_inputs, named_outputs, model, algorithm)

    def get_auxiliary_code(self):
        """
        Extra code required by cross-validation
        """
        if self.parameters.get('perform_cross_validation') in [True, '1', 1]:
            return ['templates/cross_validation.py']
        else:
            return []


class SvmModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = SvmClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(SvmModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class LogisticRegressionModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LogisticRegressionClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LogisticRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class DecisionTreeModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = DecisionTreeClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(DecisionTreeModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class GBTModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GBTClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GBTModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class NaiveBayesModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = NaiveBayesClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(NaiveBayesModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class RandomForestModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = RandomForestClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(RandomForestModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class PerceptronModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = PerceptronClassifier(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(PerceptronModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class OneVsRestModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = OneVsRestModelOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(OneVsRestModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


# Regression operations
class RegressionOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        model_in_ports = {
            'train input data': named_inputs.get('train input data'),
            'algorithm': 'algorithm'}

        model = RegressionModelOperation(
            parameters, model_in_ports, named_outputs)
        model.clone_algorithm = False
        super(RegressionOperation, self).__init__(
            parameters, named_inputs, named_outputs, model, algorithm)


class LinearRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LinearRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LinearRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class GeneralizedLinearRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GeneralizedLinearRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GeneralizedLinearRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class DecisionTreeRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = DecisionTreeRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(DecisionTreeRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class GBTRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GBTRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GBTRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class AFTSurvivalRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = AFTSurvivalRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(AFTSurvivalRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class RandomForestRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = RandomForestRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(RandomForestRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class IsotonicRegressionModelOperation(RegressionOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = IsotonicRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(IsotonicRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


# Clustering operations

class ClusteringOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        model_in_ports = {
            'train input data': named_inputs.get('train input data'),
            'algorithm': 'algorithm'}

        model = ClusteringModelOperation(
            parameters, model_in_ports, named_outputs)
        super(ClusteringOperation, self).__init__(
            parameters, named_inputs, named_outputs, model, algorithm)


class KMeansModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = KMeansClusteringOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(KMeansModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class KModesModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = KModesClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(KModesModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class GaussianMixtureModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GaussianMixtureClusteringOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GaussianMixtureModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)


class LDAModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LdaClusteringOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LDAModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
