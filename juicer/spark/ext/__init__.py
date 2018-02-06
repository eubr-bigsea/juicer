import string
import unicodedata

try:
    from pyspark import keyword_only
    from pyspark.ml import Transformer
    from pyspark.ml.param.shared import Param, HasOutputCol
    from pyspark.sql import functions, types
except ImportError:
    pass


def remove_punctuation_udf(text):
    return functions.udf(lambda t: t.translate(
        dict((ord(char), None) for char in string.punctuation)),
                         types.StringType())(text)


def strip_accents_udf(text):
    return functions.udf(lambda t: ''.join(
        c for c in unicodedata.normalize('NFD', t) if
        unicodedata.category(c) != 'Mn'), types.StringType())(text)


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
