# coding=utf-8
from pyspark.mllib.evaluation import BinaryClassificationMetrics


class CurveMetrics(BinaryClassificationMetrics):
    """
    Scala version implements .roc() and .pr()
    Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
    Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html
    See: https://stackoverflow.com/a/57342431/1646932
    """
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    # noinspection PyMethodMayBeStatic
    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter,
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2,
            # which doesn't appear to have a py4j mapping
            # noinspection PyProtectedMember
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)
