# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation


class FrequentItemSetOperation(Operation):
    """
    Finds frequent item sets in a data source composed by transactions.
    Current implementation uses FP-Growth.
    """
    MIN_SUPPORT_PARAM = 'min_support'
    ATTRIBUTE_PARAM = 'attribute'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.MIN_SUPPORT_PARAM not in parameters:
            raise ValueError(
                'Support must be informed for classifier {}'.format(
                    self.__class__))

        self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))

        self.output = self.named_outputs.get(
            'output data',
            'freq_items_{}'.format(self.order))

        self.attribute = parameters.get(self.ATTRIBUTE_PARAM)

        self.has_code = len(self.named_inputs) == 1

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = """
            from pyspark.mllib.fpm import FPGrowth
            from pyspark.sql.types import StructType, StructField, StringType
            # Current version of Spark supports FP-Growth only in RDD.
            # Assume that data is a line with transaction items separated by
            # space.
            data = {input}.rdd
            inx = reduce(
                lambda x, y: max(x, y),
                [inx for inx, x in enumerate({input}.schema)
                    if x.name == '{attr}'], 0)

            transactions = data.map(lambda line: line[inx].strip().split(' '))
            model = FPGrowth.train(
                transactions, minSupport={support}, numPartitions=10)

            emit_event(name='update task', message='Model trained',
                       status='RUNNING', identifier='{task_id}')

            items = model.freqItemsets()
            if items.isEmpty():
                schema = StructType(
                    [StructField("freq_item_sets", StringType(), True), ])
                {output} = spark_session.createDataFrame([], schema)
            else:
                {output} = items.toDF(['freq_item_sets'])
        """.format(input=self.named_inputs['input data'],
                   support=self.min_support,
                   output=self.output,
                   attr=self.attribute,
                   task_id=self.parameters['task']['id'])

        return dedent(code)
