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
        logger = spark_session.sparkContext._jvm.org.apache.log4j
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
