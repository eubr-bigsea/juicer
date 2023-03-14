# -*- coding: utf-8 -*-

import os

from juicer import operation
from juicer.transpiler import Transpiler

import juicer.scikit_learn.associative_operation as associative
import juicer.scikit_learn.classification_operation as classifiers
import juicer.scikit_learn.clustering_operation as clustering
import juicer.scikit_learn.data_operation as io
import juicer.scikit_learn.etl_operation as etl
import juicer.scikit_learn.feature_operation as feature_extraction
import juicer.scikit_learn.geo_operation as geo
import juicer.scikit_learn.model_operation as model
import juicer.scikit_learn.regression_operation as regression
import juicer.scikit_learn.text_operation as text_operations
import juicer.scikit_learn.vis_operation as vis_operation
import juicer.scikit_learn.outlier_detection as lof
import juicer.scikit_learn.stat_operation as stat_operation
import juicer.scikit_learn.nlp_operation as nlp_operation

import juicer.scikit_learn.polars.data_operation as polars_io
import juicer.scikit_learn.polars.etl_operation as polars_etl
import juicer.scikit_learn.polars.feature_operation as polars_feature
import juicer.scikit_learn.polars.vis_operation as polars_vis

import juicer.scikit_learn.duckdb.data_operation as duckdb_io
import juicer.scikit_learn.duckdb.etl_operation as duckdb_etl
import juicer.scikit_learn.duckdb.feature_operation as duckdb_feature

# noinspection SpellCheckingInspection


class ScikitLearnTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Scikit-Learn.
    """

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        self.variant = configuration.get('variant', 
            configuration.get('app_configs', {}).get('variant', 'pandas'))
        super().__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)
        if self.variant == 'polars':
            self._assign_polars_operations()
        elif self.variant == 'duckdb':
            self._assign_duckdb_operations()
        else:
            self._assign_operations()

    def get_context(self):
        dict_msgs = {
            'task_completed': _('Task completed'),
            'task_running': _('Task running'),
            'lemonade_task_completed': _('Lemonade task %s completed'),
            'lemonade_task_parents': _('Parents completed, submitting %s'),
            'lemonade_task_started': _('Lemonade task %s started'),
            'lemonade_task_afterbefore': _(
                "Submitting parent task {} before {}")}

        return {'dict_msgs': dict_msgs}

    def _assign_polars_operations(self):
        data_ops = {
            'data-reader': polars_io.DataReaderOperation,
            'data-writer': polars_io.SaveOperation,
            'save': polars_io.SaveOperation,
        }
        etl_ops = {
            'add-columns': polars_etl.AddColumnsOperation,
            'add-rows': polars_etl.UnionOperation,
            'aggregation': polars_etl.AggregationOperation,  # TODO: agg sem groupby
            'cast': polars_etl.CastOperation,
            'clean-missing': polars_etl.CleanMissingOperation,
            'difference': polars_etl.DifferenceOperation,
            'drop': polars_etl.DropOperation,
            'execute-python': polars_etl.ExecutePythonOperation,
            'execute-sql': polars_etl.ExecuteSQLOperation,
            'filter-selection': polars_etl.FilterOperation,
            'join': polars_etl.JoinOperation,
            'k-fold': polars_etl.SplitKFoldOperation,
            'projection': polars_etl.SelectOperation,
            'remove-duplicated-rows': polars_etl.DistinctOperation,
            'replace-value': polars_etl.ReplaceValuesOperation,
            'sample': polars_etl.SampleOrPartitionOperation,
            'set-intersection': polars_etl.IntersectionOperation,
            'sort': polars_etl.SortOperation,
            'split': polars_etl.SplitOperation,
            'transformation': polars_etl.TransformationOperation,
            # TODO in 'transformation': test others functions
            'rename-attr': polars_etl.RenameAttrOperation,
        }
        feature = {
            # ------ Feature Extraction Operations  ------#
            'feature-assembler': polars_feature.FeatureAssemblerOperation,
            'feature-disassembler':
                polars_feature.FeatureDisassemblerOperation,
            'min-max-scaler': polars_feature.MinMaxScalerOperation,
            'max-abs-scaler': polars_feature.MaxAbsScalerOperation,
            'one-hot-encoder': polars_feature.OneHotEncoderOperation,
            'pca': polars_feature.PCAOperation,
            'kbins-discretizer':
                polars_feature.KBinsDiscretizerOperation,
            'standard-scaler': polars_feature.StandardScalerOperation,
            'feature-indexer': polars_feature.StringIndexerOperation,
            'string-indexer': polars_feature.StringIndexerOperation,
            'locality-sensitive-hashing': polars_feature.LSHOperation,
        }
        visualization = {
            'visualization': polars_vis.VisualizationOperation,
        }


        self.operations = {}
        for ops in [data_ops, etl_ops, feature, visualization]:
            self.operations.update(ops)
        self._assign_common_operations()

    def _assign_duckdb_operations(self):
        data_ops = {
            'data-reader': duckdb_io.DataReaderOperation,
            'data-writer': duckdb_io.SaveOperation,
            'save': duckdb_io.SaveOperation,
        }
        etl_ops = {
            'add-columns': duckdb_etl.AddColumnsOperation,
            'add-rows': duckdb_etl.UnionOperation,
            'aggregation': duckdb_etl.AggregationOperation,  # TODO: agg sem groupby
            'cast': duckdb_etl.CastOperation,
            'clean-missing': duckdb_etl.CleanMissingOperation,
            'difference': duckdb_etl.DifferenceOperation,
            'drop': duckdb_etl.DropOperation,
            'execute-python': duckdb_etl.ExecutePythonOperation,
            'execute-sql': duckdb_etl.ExecuteSQLOperation,
            'filter-selection': duckdb_etl.FilterOperation,
            'join': duckdb_etl.JoinOperation,
            'k-fold': duckdb_etl.SplitKFoldOperation,
            'projection': duckdb_etl.SelectOperation,
            'remove-duplicated-rows': duckdb_etl.DistinctOperation,
            'replace-value': duckdb_etl.ReplaceValuesOperation,
            'sample': duckdb_etl.SampleOrPartitionOperation,
            'set-intersection': duckdb_etl.IntersectionOperation,
            'sort': duckdb_etl.SortOperation,
            'split': duckdb_etl.SplitOperation,
            'transformation': duckdb_etl.TransformationOperation,
            # TODO in 'transformation': test others functions
            'rename-attr': duckdb_etl.RenameAttrOperation,
        }
        feature = {
            # ------ Feature Extraction Operations  ------#
            'feature-assembler': duckdb_feature.FeatureAssemblerOperation,
            'feature-disassembler':
                duckdb_feature.FeatureDisassemblerOperation,
            'min-max-scaler': duckdb_feature.MinMaxScalerOperation,
            'max-abs-scaler': duckdb_feature.MaxAbsScalerOperation,
            'one-hot-encoder': duckdb_feature.OneHotEncoderOperation,
            'pca': duckdb_feature.PCAOperation,
            'kbins-discretizer':
                duckdb_feature.KBinsDiscretizerOperation,
            'standard-scaler': duckdb_feature.StandardScalerOperation,
            'feature-indexer': duckdb_feature.StringIndexerOperation,
            'string-indexer': duckdb_feature.StringIndexerOperation,
            'locality-sensitive-hashing': duckdb_feature.LSHOperation,
        }


        self.operations = {}
        for ops in [data_ops, etl_ops, feature]:
            self.operations.update(ops)
        self._assign_common_operations()

    def _assign_operations(self):
        if self.variant == 'polars':
            self._assign_polars_operations()
            return
        elif self.variant == 'duckdb':
            self._assign_duckdb_operations()
            return

        etl_ops = {
            'add-columns': etl.AddColumnsOperation,
            'add-rows': etl.UnionOperation,
            'aggregation': etl.AggregationOperation,  # TODO: agg sem groupby
            'cast': etl.CastOperation,
            'clean-missing': etl.CleanMissingOperation,
            'difference': etl.DifferenceOperation,
            'drop': etl.DropOperation,
            'execute-python': etl.ExecutePythonOperation,
            'execute-sql': etl.ExecuteSQLOperation,
            'filter-selection': etl.FilterOperation,
            'join': etl.JoinOperation,
            'k-fold': etl.SplitKFoldOperation,
            'locality-sensitive-hashing': feature_extraction.LSHOperation,
            'projection': etl.SelectOperation,
            'remove-duplicated-rows': etl.DistinctOperation,
            'replace-value': etl.ReplaceValuesOperation,
            'sample': etl.SampleOrPartitionOperation,
            'set-intersection': etl.IntersectionOperation,
            'sort': etl.SortOperation,
            'split': etl.SplitOperation,
            'transformation': etl.TransformationOperation,
            # TODO in 'transformation': test others functions
            'rename-attr': etl.RenameAttrOperation,
        }

        data_ops = {
            'data-reader': io.DataReaderOperation,
            'data-writer': io.SaveOperation,
            'save': io.SaveOperation,
            # 'change-attribute': io.ChangeAttributesOperation,
        }
        self.operations = {}
        for ops in [data_ops, etl_ops]:
            self.operations.update(ops)
        self._assign_common_operations()

    def _assign_common_operations(self):
        geo_ops = {
            'read-shapefile': geo.ReadShapefileOperation,
            'stdbscan': geo.STDBSCANOperation,
            'within': geo.GeoWithinOperation,
            'cartographic-projection': geo.CartographicProjectionOperation,
        }

        ml_ops = {
            # ------ Associative ------#
            'association-rules': associative.AssociationRulesOperation,
            'frequent-item-set': associative.FrequentItemSetOperation,
            'sequence-mining': associative.SequenceMiningOperation,

            # ------ Model Operations  ------#
            'apply-model': model.ApplyModelOperation,
            'cross-validation': model.CrossValidationOperation,
            'evaluate-model': model.EvaluateModelOperation,
            'load-model': model.LoadModel,
            'save-model': model.SaveModel,

            # ------ Clustering      -----#
            'agglomerative-clustering':
                clustering.AgglomerativeModelOperation,
            'dbscan-clustering': clustering.DBSCANClusteringModelOperation,
            'gaussian-mixture':
                clustering.GaussianMixtureClusteringModelOperation,
            'k-means-clustering-model': clustering.KMeansModelOperation,
            'lda-clustering-model': clustering.LdaClusteringModelOperation,
            'topic-report': clustering.TopicReportOperation,

            # ------ Classification  -----#
            'decision-tree-classifier-model':
                classifiers.DecisionTreeClassifierModelOperation,
            'gbt-classifier-model': classifiers.GBTClassifierModelOperation,
            'knn-classifier-model': classifiers.KNNClassifierModelOperation,
            'logistic-regression-model':
                classifiers.LogisticRegressionModelOperation,
            'mlp-classifier-model': classifiers.MLPClassifierModelOperation,
            'naive-bayes-classifier-model':
                classifiers.NaiveBayesClassifierModelOperation,
            'perceptron-classifier-model':
                classifiers.PerceptronClassifierModelOperation,
            'random-forest-classifier-model':
                classifiers.RandomForestClassifierModelOperation,
            'svm-classification-model': classifiers.SvmClassifierModelOperation,

            # ------ Regression  -----#
            'gbt-regressor-model':
                regression.GradientBoostingRegressorModelOperation,
            'generalized-linear-regression':
                regression.GeneralizedLinearRegressionModelOperation,
            'huber-regressor-model': regression.HuberRegressorModelOperation,
            'isotonic-regression-model':
                regression.IsotonicRegressionModelOperation,
            'linear-regression-model':
                regression.LinearRegressionModelOperation,
            'mlp-regressor-model': regression.MLPRegressorModelOperation,
            'random-forest-regressor-model':
                regression.RandomForestRegressorModelOperation,
            'sgd-regressor-model': regression.SGDRegressorModelOperation,

            # ------ Outlier  -----#
            'local-outlier-factor': lof.OutlierDetectionOperation,
        }

        nlp_ops = {
            'tokenize': nlp_operation.TokenizeOperation,
            'synonyms': nlp_operation.SynonymsOperation,
            'antonyms': nlp_operation.AntonymsOperation,
            'definer': nlp_operation.DefinerOperation,
            'comment': operation.NoOp,
            'stemming': nlp_operation.StemmingOperation,
            'lemmatization': nlp_operation.LemmatizationOperation,
            'normalizer': nlp_operation.NormalizationOperation,
            'postagging': nlp_operation.PosTaggingOperation,
            'wordsegmentation': nlp_operation.WordSegmentationOperation,
            'ner': nlp_operation.NerOperation,
            'nlp-word-counting': nlp_operation.WordCountingOperation,
            'nlp-lower-case': nlp_operation.LowerCaseOperation,
        }

        text_ops = {
            'generate-n-grams': text_operations.GenerateNGramsOperation,
            'remove-stop-words': text_operations.RemoveStopWordsOperation,
            'tokenizer': text_operations.TokenizerOperation,
            'word-to-vector': text_operations.WordToVectorOperation
        }

        statistical_ops = {
            'pdf': stat_operation.PdfOperation,
            'cdf': stat_operation.CdfOperation,
            'ccdf': stat_operation.CcdfOperation,
        }

        ws_ops = {}

        vis_ops = {
            'publish-as-visualization':
                vis_operation.PublishVisualizationOperation,
            'bar-chart': vis_operation.BarChartOperation,
            'donut-chart': vis_operation.DonutChartOperation,
            'pie-chart': vis_operation.PieChartOperation,
            'area-chart': vis_operation.AreaChartOperation,
            'line-chart': vis_operation.LineChartOperation,
            'table-visualization': vis_operation.TableVisualizationOperation,
            'summary-statistics': vis_operation.SummaryStatisticsOperation,
            'plot-chart': vis_operation.ScatterPlotOperation,
            'scatter-plot': vis_operation.ScatterPlotOperation,
            'map-chart': vis_operation.MapOperation,
            'map': vis_operation.MapOperation
        }

        for ops in [geo_ops, ml_ops, nlp_ops,
                    text_ops, ws_ops, statistical_ops, vis_ops]:
            self.operations.update(ops)
