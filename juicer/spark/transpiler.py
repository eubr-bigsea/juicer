# -*- coding: utf-8 -*-


import juicer.spark.advanced_operation as advanced_operation
import juicer.spark.data_operation as data_operation
import juicer.spark.data_quality_operation as data_quality_operation
import juicer.spark.dm_operation as dm_operation
import juicer.spark.etl_operation as etl_operation
import juicer.spark.feature_operation as feature_operation
import juicer.spark.geo_operation as geo_operation
import juicer.spark.ml_operation as ml_operation
import juicer.spark.ml_operation2 as ml_operation2
import juicer.spark.statistic_operation as statistic_operation
import juicer.spark.text_operation as text_operation
import juicer.spark.trustworthy_operation as trustworthy_operation
import juicer.spark.vis_operation as vis_operation
import juicer.spark.ws_operation as ws_operation
import os
from juicer import operation
from juicer.transpiler import Transpiler


class SparkTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Apache Spark.
    """

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        super(SparkTranspiler, self).__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)
        self.requires_hive = False
        self.requires_hive_warehouse = False
        self.hive_metadata = None

    def on(self, event, params):
        """ Manage events from Operations during code conversion """
        if event == 'requires-hive':
            self.requires_hive = True
            self.hive_metadata = params

    # noinspection SpellCheckingInspection
    def _assign_operations(self):
        
        advanced_ops = {
            'user-filter': advanced_operation.UserFilterOperation,
        }
        etl_ops = {
            'add-columns': etl_operation.AddColumnsOperation,
            'add-rows': etl_operation.AddRowsOperation,
            'aggregation': etl_operation.AggregationOperation,
            'clean-missing': etl_operation.CleanMissingOperation,
            'difference': etl_operation.DifferenceOperation,
            'distinct': etl_operation.RemoveDuplicatedOperation,
            'drop': etl_operation.DropOperation,
            'execute-python': etl_operation.ExecutePythonOperation,
            'execute-sql': etl_operation.ExecuteSQLOperation,
            'filter': etl_operation.FilterOperation,
            # Alias for filter
            'filter-selection': etl_operation.FilterOperation,
            'intersection': etl_operation.IntersectionOperation,
            'join': etl_operation.JoinOperation,
            # synonym for select
            'projection': etl_operation.SelectOperation,
            # synonym for distinct
            'remove-duplicated-rows': etl_operation.RemoveDuplicatedOperation,
            'replace-value': etl_operation.ReplaceValueOperation,
            'sample': etl_operation.SampleOrPartitionOperation,
            'select': etl_operation.SelectOperation,
            # synonym of intersection'
            'set-intersection': etl_operation.IntersectionOperation,
            'sliding-window': etl_operation.SlidingWindowOperation,
            'sliding_window': etl_operation.SlidingWindowOperation, # FIXME: remove
            'sort': etl_operation.SortOperation,
            'split': etl_operation.SplitOperation,
            'split-k-fold': etl_operation.SplitKFoldOperation,
            'transformation': etl_operation.TransformationOperation,
            'window-transformation':
                etl_operation.WindowTransformationOperation,
        }
        dm_ops = {
            'frequent-item-set': dm_operation.FrequentItemSetOperation,
            'association-rules': dm_operation.AssociationRulesOperation,
            'sequence-mining': dm_operation.SequenceMiningOperation,
        }
        ml_ops = {
            'apply-model': ml_operation.ApplyModelOperation,
            'classification-model': ml_operation.ClassificationModelOperation,
            'classification-report': ml_operation.ClassificationReport,
            'clustering-model': ml_operation.ClusteringModelOperation,
            'cross-validation': ml_operation.CrossValidationOperation,
            'decision-tree-classifier':
                ml_operation.DecisionTreeClassifierOperation,
            'one-vs-rest-classifier': ml_operation.OneVsRestClassifier,
            'evaluate-model': ml_operation.EvaluateModelOperation,
            'feature-assembler': ml_operation.FeatureAssemblerOperation,
            'feature-disassembler': ml_operation.FeatureDisassemblerOperation,
            'vector-indexer': ml_operation.VectorIndexOperation,
            'feature-indexer': ml_operation.StringIndexerOperation,
            'gaussian-mixture-clustering':
                ml_operation.GaussianMixtureClusteringOperation,
            'gbt-classifier': ml_operation.GBTClassifierOperation,
            'isotonic-regression': ml_operation.IsotonicRegressionOperation,
            'k-means-clustering': ml_operation.KMeansClusteringOperation,
            'lda-clustering': ml_operation.LdaClusteringOperation,
            'lsh': ml_operation.LSHOperation,
            'naive-bayes-classifier':
                ml_operation.NaiveBayesClassifierOperation,
            'one-hot-encoder': ml_operation.OneHotEncoderOperation,
            'pca': ml_operation.PCAOperation,
            'perceptron-classifier': ml_operation.PerceptronClassifier,
            'random-forest-classifier':
                ml_operation.RandomForestClassifierOperation,
            'svm-classification': ml_operation.SvmClassifierOperation,
            'topic-report': ml_operation.TopicReportOperation,
            'recommendation-model': ml_operation.RecommendationModel,
            'als-recommender': ml_operation.AlternatingLeastSquaresOperation,
            'logistic-regression':
                ml_operation.LogisticRegressionClassifierOperation,
            'linear-regression': ml_operation.LinearRegressionOperation,
            'regression-model': ml_operation.RegressionModelOperation,
            'index-to-string': ml_operation.IndexToStringOperation,
            'random-forest-regressor':
                ml_operation.RandomForestRegressorOperation,
            'gbt-regressor': ml_operation.GBTRegressorOperation,
            'generalized-linear-regressor':
                ml_operation.GeneralizedLinearRegressionOperation,
            'aft-survival-regression':
                ml_operation.AFTSurvivalRegressionOperation,
            'save-model': ml_operation.SaveModelOperation,
            'load-model': ml_operation.LoadModelOperation,
            'voting-classifier': ml_operation.VotingClassifierOperation,
            'outlier-detection': ml_operation.OutlierDetectionOperation,
        }

        data_ops = {
            'change-attribute': data_operation.ChangeAttributeOperation,
            'data-reader': data_operation.DataReaderOperation,
            'data-source': data_operation.DataSourceOperation,
            'data-writer': data_operation.SaveOperation,
            'external-input': data_operation.ExternalInputOperation,
            'read-csv': data_operation.ReadCSVOperation,
            'save': data_operation.SaveOperation,
        }
        data_quality_ops = {
            'entity-matching': data_quality_operation.EntityMatchingOperation,
        }
        statistics_ops = {
            'kaplan-meier-survival':
                statistic_operation.KaplanMeierSurvivalOperation,
            'cox-proportional-hazards':
                statistic_operation.CoxProportionalHazardsOperation
        }
        other_ops = {
            'comment': operation.NoOp,
        }
        geo_ops = {
            'read-shapefile': geo_operation.ReadShapefile,
            'within': geo_operation.GeoWithin,
        }
        text_ops = {
            'generate-n-grams': text_operation.GenerateNGramsOperation,
            'remove-stop-words': text_operation.RemoveStopWordsOperation,
            'tokenizer': text_operation.TokenizerOperation,
            'word-to-vector': text_operation.WordToVectorOperation
        }
        ws_ops = {
            'multiplexer': ws_operation.MultiplexerOperation,
            'service-output': ws_operation.ServiceOutputOperation,

        }
        vis_ops = {
            'area-chart': vis_operation.AreaChartOperation,
            'bar-chart': vis_operation.BarChartOperation,
            'box-plot': vis_operation.BoxPlotOperation,
            'bubble-chart': vis_operation.BubbleChartOperation,
            'donut-chart': vis_operation.DonutChartOperation,
            'heatmap': vis_operation.HeatmapOperation,
            'histogram': vis_operation.HistogramOperation,
            'iframe': vis_operation.IFrameOperation,
            'indicator': vis_operation.IndicatorOperation,
            'line-chart': vis_operation.LineChartOperation,
            'map-chart': vis_operation.MapOperation,
            'map': vis_operation.MapOperation,
            'markdown': vis_operation.MarkdownOperation,
            'pie-chart': vis_operation.PieChartOperation,
            'plot-chart': vis_operation.ScatterPlotOperation,
            'publish-as-visualization':
                vis_operation.PublishVisualizationOperation,
            'scatter-plot': vis_operation.ScatterPlotOperation,
            'summary-statistics': vis_operation.SummaryStatisticsOperation,
            'table-visualization': vis_operation.TableVisualizationOperation,
            'treemap': vis_operation.TreemapOperation,
        }
        feature_ops = {
            'bucketizer': feature_operation.BucketizerOperation,
            'chi-sq-selector':
                feature_operation.ChiSquaredSelectorOperation,
            'quantile-discretizer':
                feature_operation.QuantileDiscretizerOperation,
            'standard-scaler': feature_operation.StandardScalerOperation,
            'max-abs-scaler': feature_operation.MaxAbsScalerOperation,
            'min-max-scaler': feature_operation.MinMaxScalerOperation,

        }
        trustworthy_operations = {
            'fairness-evaluator':
                trustworthy_operation.FairnessEvaluationOperation
        }
        ml_model_operations2 = {
            'decision-tree-classifier-model':
                ml_operation2.DecisionTreeModelOperation,
            'logistic-regression-classifier-model':
                ml_operation2.LogisticRegressionModelOperation,
            'gbt-classifier-model': ml_operation2.GBTModelOperation,
            'naive-bayes-classifier-model':
                ml_operation2.NaiveBayesModelOperation,
            'random-forest-classifier-model':
                ml_operation2.RandomForestModelOperation,
            'perceptron-classifier-model':
                ml_operation2.PerceptronModelOperation,
            'one-vs-rest-classifier-model-':
                ml_operation2.OneVsRestModelOperation,
            'svm-classification-model': ml_operation2.SvmModelOperation,

            'k-means-clustering-model': ml_operation2.KMeansModelOperation,
            'k-modes-clustering-model': ml_operation2.KModesModelOperation,
            'gaussian-mixture-clustering-model':
                ml_operation2.GaussianMixtureModelOperation,
            'lda-clustering-model': ml_operation2.LDAModelOperation,

            'decision-tree-regression-model':
                ml_operation2.DecisionTreeRegressionModelOperation,

            'isotonic-regression-model':
                ml_operation2.IsotonicRegressionModelOperation,
            'aft-survival-regression-model':
                ml_operation2.AFTSurvivalRegressionModelOperation,
            'gbt-regressor-model':
                ml_operation2.GBTRegressionModelOperation,
            'random-forest-regressor-model':
                ml_operation2.RandomForestRegressionModelOperation,
            'generalized-linear-regressor-model':
                ml_operation2.GeneralizedLinearRegressionModelOperation,
            'linear-regression-model':
                ml_operation2.LinearRegressionModelOperation,

        }

        self.operations = {}
        for ops in [advanced_ops, data_ops, etl_ops, geo_ops, ml_ops, other_ops, 
                    text_ops, statistics_ops, ws_ops, vis_ops, dm_ops, 
                    data_quality_ops, feature_ops, trustworthy_operations, 
                    ml_model_operations2]:
            self.operations.update(ops)
