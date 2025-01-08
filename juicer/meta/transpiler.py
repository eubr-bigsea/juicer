# -*- coding: utf-8 -*-

import ast
from dataclasses import dataclass
import os
from typing import Callable
from gettext import gettext
import uuid
import juicer.meta.operations as ops
from collections import namedtuple
from juicer.service import tahiti_service
from juicer.transpiler import Transpiler

class SqlWorkflowTemplateParams:
    def __init__(self, readers=None, cells=None):
        self.readers: list[ops.DataReaderOperation] = sorted(
            readers, key=lambda x: x.task.get('display_order'))
        self.cells: list[ops.ExecuteSQLOperation] = sorted(
            cells, key=lambda x: x.task.get('display_order'))


class ModelBuilderTemplateParams:
    __all__ = ('evaluator', 'estimators', 'grid', 'read_data', 'sample',
               'reduction', 'split', 'features', 'enabled')
    def __init__(self, evaluator=None, estimators=None, grid=None,
                 read_data=None, sample=None, reduction=None, split=None,
                 features=None, save_data=None, save_model=None):
        self.evaluator: ops.EvaluatorOperation = evaluator
        self.estimators: list[ops.EstimatorMetaOperation] = estimators
        self.grid: ops.GridOperation = grid
        self.read_data: ops.ReadDataOperation = read_data
        self.sample: ops.SampleOperation = sample
        self.reduction: ops.FeaturesReductionOperation = reduction
        self.split: ops.SplitOperation = split
        self.features: ops.FeaturesOperation = features
        self.save_data = save_data
        self.save_model = save_model


# noinspection SpellCheckingInspection

# https://github.com/microsoft/pylance-release/issues/140#issuecomment-661487878
_: Callable[[str], str]


class MetaTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) to code in
    Meta JSON format and then to the final (Python) target platform.
    """

    SUPPORTED_TARGET_PLATFORMS = {
        'spark': 1,
        'scikit-learn': 4
    }

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        super(MetaTranspiler, self).__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)

        self.target_platform = 'spark'
        self._assign_operations()

    def get_context(self) -> dict:
        """Returns extra variables to be used in template

        Returns:
            _type_: Dict with extra variables
        """
        return {'target_platform_id':
                self.SUPPORTED_TARGET_PLATFORMS.get(
                    self.target_platform, 1),
                'target_platform': self.target_platform or 'spark'}

    def _assign_operations(self):
        self.operations = {
            'execute-sql': ops.ExecuteSQLOperation,
            'execute-python': ops.ExecutePythonOperation,
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
            'one-hot-encoding': ops.OneHotEncodingOperation,
            'string-indexer': ops.StringIndexerOperation,

            'n-grams': ops.GenerateNGramsOperation,
            'remove-missing': ops.RemoveMissingOperation,
            'force-range': ops.ForceRangeOperation,
        }
        transform = [
            'extract-numbers',
            'extract-with-regex',
            'replace-with-regex',
            'to-upper', 'to-lower', 'initcap', 'capitalize', 'remove-accents',
            'split-into-words',
            'trim', 'normalize-text',
            'truncate-text',
            'parse-to-date',

            'round-number',
            'ts-to-date',

            'date-to-ts',
            'date-part',
            'date-add',
            'format-date',
            'truncate-date-to',

            'invert-boolean',

            'extract-from-array',
            'concat-array',
            'sort-array',
            'change-array-type',

            'flag-empty',
            'flag-with-formula',
        ]
        model = {
            'evaluator': ops.EvaluatorOperation,
            'features': ops.FeaturesOperation,
            'features-reduction': ops.FeaturesReductionOperation,
            'split': ops.SplitOperation,
            # 'bucketize': ops.BucketizeOperation,
            'rescale': ops.RescaleOperation,
            'grid': ops.GridOperation,
            'k-means': ops.KMeansOperation,
            'gaussian-mix': ops.GaussianMixOperation,
            'pic_clustering':ops.PowerIterationClusteringOperation,
            'lda_clustering':ops.LDAOperation,
            'bkm_clustering':ops.BisectingKMeansOperation,
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
            'generalized-linear-regressor':
                ops.GeneralizedLinearRegressionOperation,
            'decision-tree-regressor': ops.DecisionTreeRegressorOperation,
            'fm-classifier': ops.FactorizationMachinesClassifierOperation,
            'fm-regression': ops.FactorizationMachinesRegressionOperation,
        }


        self.operations.update(model)

        visualizations = {'visualization': ops.VisualizationOperation}
        self.operations.update(visualizations)

        batch= {
            'convert-data-source': ops.ConvertDataSourceFormat
        }
        self.operations.update(batch)

        for f in transform:
            self.operations[f] = ops.TransformOperation
    def prepare_model_builder_parameters(self, tasks, workflow_forms=None,
        user=None) -> \
            ModelBuilderTemplateParams:
        """ Organize operations to be used in the code generation
        template.

        Args:
            ops (list): List of operations

        Returns:
            _type_: Model builder parameters
        """
        if workflow_forms is None:
            workflow_forms = {}
        estimators = {'k-means', 'gaussian-mix', 'pic_clustering',
                      'lda_clustering', 'bkm_clustering',
                      'decision-tree-classifier',
                      'gbt-classifier', 'naive-bayes', 'perceptron',
                      'random-forest-classifier', 'logistic-regression', 'svm',
                      'linear-regression', 'isotonic-regression',
                      'gbt-regressor', 'random-forest-regressor',
                      'generalized-linear-regressor', 'decision-tree-regressor',
                      'fm-classifier', 'fm-regression'}
        param_dict = {'estimators': []}
        for task in tasks:
            slug = task.task.get('operation').get('slug')
            if slug == 'read-data':
                param_dict['read_data'] = task
            elif slug == 'features-reduction':
                param_dict['reduction'] = task
            elif slug in estimators:
                param_dict['estimators'].append(task)
            else:
                param_dict[slug] = task

        def get_value(name, default=None):
            return workflow_forms.get(name, {"value": default}).get("value")

        save_data_task = None
        from juicer.spark.data_operation import SaveOperation
        if bool(get_value("save_data", False)):
            save_data_task = SaveOperation(
                {
                    "task": {"id": str(uuid.uuid4())},
                    "name": get_value("data_name"),
                    "path": get_value("data_path"),
                    "storage": get_value("data_storage"),
                    "format": "PARQUET",
                    "mode": get_value("data_overwrite"),
                    "configuration": self.configuration,
                    "use_storage_path": True,
                    "user": user,
               },
                {"input data": "df"},
                {},
            )
        param_dict["save_data"] = save_data_task # type: ignore

        from juicer.spark.ml_operation import SaveModelOperation
        if bool(get_value("save_model", False)):
            save_model_task = SaveModelOperation(
                {
                    "task": {"id": str(uuid.uuid4())},
                    "name": get_value("data_name"),
                    "path": get_value("data_path"),
                    "storage": get_value("data_storage"),
                    "format": "DEFAULT",
                    "mode": get_value("data_overwrite"),
                    "configuration": self.configuration,
                    "use_storage_path": True,
                    "user": user,
               },
                {"models": "model"},
                {},
            )
        param_dict["save_model"] = save_model_task # type: ignore

        # save_model_task = None
        # if bool(get_value("save_model", False)):
        #     save_model_task = ops.SaveOperation(
        #         {
        #             "name": get_value("name"),
        #             "path": get_value("path"),
        #             "storage": get_value("storage"),
        #             "format": get_value("format"),
        #             "tags": get_value("tags"),
        #             "mode": get_value("mode"),
        #             "header": get_value("header"),
        #         },
        #         {},
        #         {},
        #     )
        # param_dict["save_model"] = save_model_task # type: ignore
        if (not param_dict.get('estimators')):
            raise ValueError(gettext('No algorithm or algorithm parameter informed.'))
        return ModelBuilderTemplateParams(**param_dict)

    def prepare_sql_workflow_parameters(self, ops) -> \
            SqlWorkflowTemplateParams:
        """ Organize operations to be used in the code generation
        template.

        Args:
            ops (list): List of operations

        Returns:
            _type_: Sql Workflow parameters
        """
        param_dict = {'readers': [], 'cells': []}
        for op in ops:
            slug = op.task.get('operation').get('slug')
            if slug == 'read-data':
                param_dict['readers'].append(op)
            #elif slug == 'execute-sql':
            else:
                param_dict['cells'].append(op)
        return SqlWorkflowTemplateParams(**param_dict)

    def get_source_code_library_code(self, workflow):
        """Retrieve the source code stored as a code library"""
        result = []
        ids = workflow.get('forms').get(
            'code_libraries', {}).get('value', [])

        if ids:
            libraries = self._get_code_libraries(ids)
            for library in libraries.get('data'):
                code = library.get("code", "")
                self._validate_code_library(code)
                result.append(code)
        return ["\n".join(result)]

    def _validate_code_library(self, code):
        """Validate code library"""
        try:
            tree = ast.parse(code)
            # Test the code is a function definition
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    raise ValueError(
                        gettext(
                            "Provided Python code is not a function definition. "
                            "Code libraries must define Python functions."
                        )
                    )
            return True
        except SyntaxError as se:
            raise ValueError(
                gettext(
                    "Provided Python code has syntax error(s): "
                    "{text}, line {l}, offset: {o} "
                ).format(text=se.text, l=se.lineno, o=se.offset)
            )

    def _get_code_libraries(self, ids):
        tahiti_config = self.configuration["juicer"]["services"]["tahiti"]
        url = tahiti_config["url"]
        token = str(tahiti_config["auth_token"])

        ids = ",".join([str(i) for i in ids])
        return tahiti_service.query_tahiti(
            base_url=url,
            item_path="/source-codes",
            item_id=None,
            token=token,
            qs=f"ids={ids}",
        )
