# -*- coding: utf-8 -*-

import os
import juicer.meta.operations as ops
from collections import namedtuple
from juicer import operation
from juicer.transpiler import Transpiler

ModelBuilderTemplateParams = namedtuple('ModelBuilderTemplateParams',
    ['evaluator', 'estimators', 'grid', 'read_data', 'sample', 
    'reduction', 'split', 'features'])

# noinspection SpellCheckingInspection
class MetaTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Meta and then the final target platform.
    """

    SUPPORTED_TARGET_PLATFORMS = {
        'spark': 1,
        'scikit-learn': 4
    }
    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        super(MetaTranspiler, self).__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)

        self.target_platform = 'scikit-learn'
        self._assign_operations()

    def get_context(self):
        return {'target_platform_id': 
            self.SUPPORTED_TARGET_PLATFORMS.get(
                self.target_platform, 4),
            'target_platform': self.target_platform or 'scikit-learn'}

    def _assign_operations(self):
        self.operations = {
            'add-by-formula': ops.AddByFormulaOperation,
            'cast': ops.CastOperation,
            'clean-missing': ops.CleanMissingOperation,
            'concat-rows': ops.ConcatRowsOperation,
            'date-diff': ops.DateDiffOperation,
            'discard': ops.DiscardOperation,
            'duplicate': ops.DuplicateOperation,
            # 'extract-from-array': ops.ExtractFromArrayOperation,
            'filter': ops.FilterOperation,
            'find-replace': ops.FindReplaceOperation,
            'group':  ops.GroupOperation,
            'join': ops.JoinOperation,
            'read-data': ops.ReadDataOperation,
            'rename': ops.RenameOperation,
            'sample': ops.SampleOperation,
            'save': ops.SaveOperation,
            'select': ops.SelectOperation,
            'sort': ops.SortOperation,
        }
        transform = [
            'extract-numbers',
            'to-upper', 'to-lower', 'initcap', 'capitalize', 'remove-accents',
            'split-into-words',
            'trim', 'normalize-text',
            'truncate-text',

            'round-number',
            'ts-to-date',

            'date-to-ts',
            'format-date',

            'invert-boolean',

            'extract-from-array',

        ]
        model = {
            'evaluator': ops.EvaluatorOperation,
            'features': ops.FeaturesOperation,
            'features-reduction': ops.FeaturesReductionOperation,
            'split': ops.SplitOperation,
            'grid': ops.GridOperation,
            'k-means': ops.KMeansOperation,
            'gaussian-mix': ops.GaussianMixOperation,
            'decision-tree-classifier': ops.DecisionTreeClassifierOperation,
            'gbt-classifier': ops.GBTClassifierOperation,
            'naive-bayes': ops.NaiveBayesClassifierOperation,
            'perceptron': ops.PerceptronClassifierOperation,
            'random-forest-classifier': ops.RandomForestClassifierOperation,
            'logistic-regression': ops.LogisticRegressionOperation,
            'svm': ops.SVMClassifierOperation,
            'linear-regression': ops.LinearRegressionOperation,
            'isotonic-regression': ops.IsotonicRegressionOperation,
            'gbt-regressor': ops.GBTRegressorOperation,
            'random-forest-regressor': ops.RandomForestRegressorOperation,
            'generalized-linear-regressor': ops.GeneralizedLinearRegressionOperation,
            'decision-tree-regressor': ops.DecisionTreeRegressorOperation,
        }

        self.operations.update(model)

        visualizations = {'visualization': ops.VisualizationOperation}
        self.operations.update(visualizations)

        for f in transform:
            self.operations[f] = ops.TransformOperation

    def prepare_model_builder_parameters(self, ops):
        """ Organize operations to be used in the code generation
        template. """

        estimators = {'k-means', 'gaussian-mix', 'decision-tree-classifier', 
            'gbt-classifier', 'naive-bayes', 'perceptron', 
            'random-forest-classifier', 'logistic-regression', 'svm', 
            'linear-regression', 'isotonic-regression', 'gbt-regressor', 
            'random-forest-regressor', 'generalized-linear-regressor',
            'decision-tree-regressor'}

        param_dict = {'estimators': []}
        for op in ops:
            slug = op.task.get('operation').get('slug')
            if slug == 'read-data':
                param_dict['read_data'] = op
            elif slug == 'split':
                param_dict['split'] = op
            elif slug == 'sample':
                param_dict['sample'] = op
            elif slug == 'grid':
                param_dict['grid'] = op
            elif slug == 'evaluator':
                param_dict['evaluator'] = op
            elif slug == 'features':
                param_dict['features'] = op
            elif slug == 'features-reduction':
                param_dict['reduction'] = op
            elif slug in estimators:
                param_dict['estimators'].append(op)


        return ModelBuilderTemplateParams(**param_dict)

