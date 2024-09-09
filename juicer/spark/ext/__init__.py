# coding=utf-8

import string
import unicodedata

from pyspark.ml.util import JavaMLWritable, JavaMLReadable
from pyspark.ml.wrapper import JavaTransformer

from juicer.util import dataframe_util

try:
    from pyspark import keyword_only
    from pyspark.ml import Transformer
    from pyspark.ml.param.shared import Param, HasOutputCol, HasFeaturesCol, \
        HasPredictionCol, Params, TypeConverters
    from pyspark.sql import functions, types
except ImportError:
    pass


def remove_punctuation_udf(text):
    return functions.udf(lambda t: t.translate(
        dict((ord(char), None) for char in string.punctuation)) if t else t,
                         types.StringType())(text)


def strip_accents_udf(text):
    return functions.udf(lambda t: ''.join(
        c for c in unicodedata.normalize('NFD', t) if
        unicodedata.category(c) != 'Mn'), types.StringType())(text)


def ith_function_udf(v, i):
    def ith(v_, i_):
        return v_.values.item(i_)

    return functions.udf(ith, types.FloatType())(v, i)


def translate_function_udf(v, missing='null', _type='string', pairs=None):
    if pairs is None:
        pairs = [[]]

    lookup = dict(pairs)
    if _type == 'float':
        t = float
    elif _type == 'int':
        t = types.IntegerType
    elif _type == 'long':
        t = int
    elif _type == 'timestamp':
        t = types.TimestampType
    elif _type == 'string':
        t = bytes
    else:
        raise ValueError('Invalid type: {}'.format(_type))

    def translate(v_):
        if missing == 'null':
            return lookup.get(v_)
        else:
            return lookup.get(v_, v_)

    return functions.udf(translate, t())(v)


# noinspection PyPep8Naming
class CustomExpressionTransformer(Transformer, HasOutputCol):
    """
    Implements a transformer from an expression, so it enable Spark pipeline
    inclusion of operations such as TransformOperation.
    """

    @keyword_only
    def __init__(self, outputCol=None, expression=None):
        super(CustomExpressionTransformer, self).__init__()

        self.expression = Param(self, "expression", "")
        self._setDefault(expression=None)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, outputCol=None, expression=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setExpression(self, value):
        self._paramMap[self.expression] = value
        return self

    def getExpression(self):
        return self.getOrDefault(self.expression)

    def _transform(self, dataset):
        expression = self.getExpression()
        out_col = self.getOutputCol()

        return dataset.withColumn(out_col, expression)


# We need the following code in order to integrate this application log with the
# spark standard log4j. Thus, we set *__SPARK_LOG__* at most one time per
# execution.
__SPARK_LOG__ = None


def spark_logging(spark_session):
    global __SPARK_LOG__
    if not __SPARK_LOG__:
        # noinspection PyProtectedMember
        logger = spark_session._jvm.org.apache.log4j
        __SPARK_LOG__ = logger.LogManager.getLogger(__name__)
    return __SPARK_LOG__


def take_sample(df, size=100, default=None):
    """
    Takes a sample from data frame.
    """
    result = default or []
    if hasattr(df, 'take'):
        header = ','.join([f.name for f in df.schema.fields])
        result = [header]
        result.extend(
            df.limit(size).rdd.map(dataframe_util.convert_to_csv).collect())
    return result


def juicer_debug(spark_session, name, variable, data_frame):
    """ Debug code """
    spark_logging(spark_session).debug('#' * 20)
    spark_logging(spark_session).debug('|| {} ||'.format(name))
    spark_logging(spark_session).debug('== {} =='.format(variable))
    data_frame.show()
    schema = data_frame.schema
    for attr in schema:
        spark_logging(spark_session).debug('{} {} {} {}'.format(
            attr.name, attr.dataType, attr.nullable, attr.metadata))


# noinspection PyPep8Naming
# noinspection PyUnusedLocal,PyProtectedMember
class LocalOutlierFactor(JavaTransformer, HasFeaturesCol, HasOutputCol,
                         JavaMLWritable, JavaMLReadable):
    """
    Wrapper for Spark-LOF implementation
    """
    minPts = Param(Params._dummy(), "minPts", "number of points",
                   typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, featuresCol="features", outputCol="outliers",
                 minPts=2):
        super(LocalOutlierFactor, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.outlier.LOF",
                                            self.uid)
        self._setDefault(minPts=5, featuresCol="features",
                         outputCol="outliers")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, minPts=5):
        """
        setParams(self, minPts=5)
        Sets params for this LocalOutlierFactor.
        """
        return self._set(minPts=minPts)

    def setMinPts(self, value):
        """
        Sets the value of :py:attr:`minPts`.
        """
        return self._set(minPts=value)

    def getMinPts(self):
        """
        Gets the value of minPts or its default value.
        """
        return self.getOrDefault(self.minPts)


# noinspection PyPep8Naming
# noinspection PyUnusedLocal,PyProtectedMember
class FairnessEvaluatorTransformer(JavaTransformer, JavaMLWritable,
                                   JavaMLReadable):
    """
    Wrapper for Spark-LOF implementation
    """
    tau = Param(Params._dummy(), "tau", "tau",
                typeConverter=TypeConverters.toFloat)

    sensitiveColumn = Param(Params._dummy(), "sensitiveColumn",
                            "sensitiveColumn",
                            typeConverter=TypeConverters.toString)

    labelColumn = Param(Params._dummy(), "labelColumn", "labelColumn",
                        typeConverter=TypeConverters.toString)

    scoreColumn = Param(Params._dummy(), "scoreColumn", "scoreColumn",
                        typeConverter=TypeConverters.toString)

    baselineValue = Param(Params._dummy(), "baselineValue", "baselineValue",
                          typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, tau=.8, sensitiveColumn='sensitive', labelColumn='label',
                 scoreColumn='score', baselineValue=None):
        super(FairnessEvaluatorTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "br.ufmg.dcc.speed.lemonade.fairness.FairnessEvaluatorTransformer",
            self.uid)
        self._setDefault(tau=.8, sensitiveColumn='sensitive',
                         labelColumn='label',
                         scoreColumn='score', baselineValue=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, tau=.8, sensitiveColumn='sensitive',
                  labelColumn='label',
                  scoreColumn='score', baselineValue=None):
        return self._set(tau=tau, sensitiveColumn=sensitiveColumn,
                         labelColumn=labelColumn, scoreColumn=scoreColumn,
                         baselineValue=baselineValue)

 
    def setMinPts(self, value):
        """
        Sets the value of :py:attr:`minPts`.
        """
        return self._set(minPts=value)

    def getMinPts(self):
        """
        Gets the value of minPts or its default value.
        """
        return self.getOrDefault(self.minPts)


class FairnessEvaluatorSql: 
    """
      Fairness evaluation using SQL    
    """
    TABLE          = 'compas' 

    def __init__(self, sensitive_column, score_column, label_column, baseline_column, range_column=[0.8,1.25]): 
        self.sensitive_column      = sensitive_column  
        self.score_column          = score_column 
        self.label_column          = label_column 
        self.baseline_column       = baseline_column
        self.range_column          = range_column 

    def get_fairness_sql(self):
        sql = f'''
	       WITH FIRST_LEVEL AS (
  	       SELECT {self.sensitive_column}, SUM(CASE 
                    WHEN  {self.label_column}=1 THEN 1
                    ELSE 0
                 END) AS positive,
               SUM(CASE 
                    WHEN {self.label_column}=0 THEN 1
                    ELSE 0
                 END) AS negative,
               SUM(CASE 
                    WHEN {self.score_column}=1 THEN 1
                    ELSE 0
                 END) AS predicted_positive,
               SUM(CASE 
                    WHEN {self.score_column}=0 THEN 1
                    ELSE 0
                 END) AS predicted_negative, 
                SUM(CASE 
                    WHEN {self.label_column}=1 THEN 1
                    ELSE 0
                 END) AS group_label_positive,
               SUM(CASE 
                    WHEN {self.label_column}=0 THEN 1
                    ELSE 0
                 END) AS group_label_negative,  
  			   SUM(CASE 
                    WHEN {self.label_column}=0 AND {self.score_column}=0 THEN 1
                    ELSE 0
                 END) AS true_negative, 
               SUM(CASE 
                    WHEN {self.label_column}=0 AND {self.score_column}=1 THEN 1
                    ELSE 0
                 END) AS false_positive, 
               SUM(CASE 
                    WHEN {self.label_column}=1 AND {self.score_column}=0 THEN 1
                    ELSE 0
                 END) AS false_negative,
               SUM(CASE  
                    WHEN {self.label_column}=1 AND {self.score_column}=1 THEN 1 
                    ELSE 0
                 END) AS true_positive, 
               COUNT({self.sensitive_column}) AS group_size, 
	       '{self.sensitive_column}' AS attribute
  	       FROM {self.TABLE} GROUP BY {self.sensitive_column} 
	      ),
	      TOTAL_RECORDS AS (
                SELECT SUM(group_size) AS total_records FROM FIRST_LEVEL
	      ),
              TOTAL_PREDICTED_POSITIVE AS (
  	        SELECT SUM(predicted_positive) AS total_predicted_positive FROM FIRST_LEVEL
	      ),
	      SECOND_LEVEL AS (
  	        SELECT *, 
                       (true_positive)/(true_positive + false_negative) AS tpr,  -- Total positive rate or recall
       	               (false_negative)/(false_negative +true_negative) AS for,  -- for: false_omission_rate
                       (false_positive)/(predicted_positive) AS fdr,             -- fdr: false_discovery_rate
                       (false_positive)/(false_positive + true_negative) AS fpr, -- fpr: false_positive_rate 
                       (false_negative)/(false_negative + true_positive) AS fnr, -- fnr: false_negative_rate
                       (true_negative)/(true_negative + false_positive) AS tnr   -- tnr: true_negative_rate 
  	        FROM FIRST_LEVEL
	      ),  
	      THIRDTH_LEVEL AS (
  	        SELECT F.*,
                       (F.predicted_positive/PP.total_predicted_positive) AS pred_pos_ratio_k, -- predicted_positive_rate_k
                       (F.predicted_positive/F.group_size) AS pred_pos_ratio_g,                -- predicted_positive_rate_g
                       (F.true_negative)/(F.true_negative + F.false_negative) AS pred_neg,     -- predicted_negative
                       (F.true_positive)/(F.true_positive + F.false_positive) AS pred_pos      -- predicted_positive
  	        FROM SECOND_LEVEL F 
  	        CROSS JOIN TOTAL_PREDICTED_POSITIVE PP
	      ),
	      GROUP_DISPARITY AS (
  	        SELECT * FROM THIRDTH_LEVEL WHERE {self.sensitive_column}='{self.baseline_column}'
	      ),	
	      FOURTH_LEVEL AS (
	        SELECT F.*, 
	               (F.fdr/D.fdr) AS fdr_disparity,
	               (F.fnr/D.fnr) AS fnr_disparity,
	               (F.for/D.for) AS for_disparity, 
	               (F.fpr/D.fpr) AS fpr_disparity,
	               (F.pred_pos_ratio_k/D.pred_pos_ratio_k) AS pred_pos_ratio_k_disparity,
	               (F.pred_pos_ratio_g/D.pred_pos_ratio_g) AS pred_pos_ratio_g_disparity
	       FROM THIRDTH_LEVEL F 
	       CROSS JOIN GROUP_DISPARITY D 
	     ),
	     FIFTH_LEVEL AS (
  	       SELECT *,
         	      (true_positive + true_negative)/(true_positive + true_negative + false_negative + false_positive) AS accuracy,
         	      (2*true_positive)/(2*true_positive + false_positive + false_negative) AS f1_score,
         	      (positive)/(positive + negative) AS prev -- prevalence
  	       FROM FOURTH_LEVEL  
	     ),
	     SIXTH_LEVEL AS (
  	       SELECT *,
         	      sqrt(false_positive) /(sqrt(true_positive) + sqrt(false_positive)) AS prev_threshold, -- prevalence_threshold 
         	      (true_positive + true_negative - 1) AS informedness, 
         	      (pred_pos + pred_neg - 1) AS markedness, 
         	      (true_positive / false_positive) AS pos_likelihood_ratio, -- positive_likelihood_ratio
         	      (false_negative / true_negative) AS neg_likelihood_ratio,  -- neg_likelihood_ratio
         	      (true_positive / (true_positive + false_negative + false_positive)) AS jaccard_index, 
         	      sqrt(pred_pos * true_positive) AS fowlkes_mallows_index,
         	      ((true_positive * true_negative) - (false_positive * false_negative)) / 
         	      sqrt((true_positive + false_positive) * (true_positive + false_negative) *
             	      (true_negative + false_positive) * (true_negative + false_negative) ) AS matthews_correlation_coefficient
 	        FROM FIFTH_LEVEL 
	     ),
	     SEVENTH_LEVEL AS (
  		SELECT *, 
         	      (pos_likelihood_ratio / neg_likelihood_ratio) AS dor -- diagnostic_odds_ratio
  		FROM SIXTH_LEVEL ORDER BY {self.sensitive_column}
	     ),
             FAIRNESS_EVALUATION AS (
  	       SELECT *, CASE 
                   	    WHEN fdr_disparity > {self.range_column[0]} AND 
                                 fdr_disparity < {self.range_column[1]} THEN 'True' ELSE 'False'
               		 END AS fdr_parity, 
               		 CASE 
                   	    WHEN fnr_disparity > {self.range_column[0]} AND 
                                 fnr_disparity < {self.range_column[1]} THEN 'True' ELSE 'False'
               		 END AS fnr_parity,
               		 CASE 
                   	    WHEN for_disparity > {self.range_column[0]} AND 
                                 for_disparity < {self.range_column[1]} THEN 'True' ELSE 'False'
               		 END AS for_parity,
               		 CASE 
                   	    WHEN fpr_disparity > {self.range_column[0]} AND 
                                 fpr_disparity < {self.range_column[1]} THEN 'True' ELSE 'False'
              		 END AS fpr_parity, 
               		 CASE 
                   	    WHEN pred_pos_ratio_k_disparity > {self.range_column[0]} AND 
                                 pred_pos_ratio_k_disparity < {self.range_column[1]} THEN 'True' ELSE 'False'
                         END AS pred_pos_ratio_k_parity, 
                         CASE 
                            WHEN pred_pos_ratio_g_disparity > {self.range_column[0]} AND 
                                 pred_pos_ratio_g_disparity < {self.range_column[1]} THEN 'True' ELSE 'False'
                         END AS pred_pos_ratio_g_parity
               FROM SEVENTH_LEVEL 
            )
            SELECT TR.*, M.*
	    FROM FAIRNESS_EVALUATION M
            CROSS JOIN TOTAL_RECORDS TR ORDER BY {self.sensitive_column}
           '''    
                
        return sql   
