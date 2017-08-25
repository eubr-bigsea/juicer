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
    CONFIDENCE_PARAM = 'confidence'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.MIN_SUPPORT_PARAM not in parameters:
            raise ValueError(
                'Support must be informed for classifier {}'.format(
                    self.__class__))

        self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))

        self.output = self.named_outputs.get(
            'output data', 'freq_items_{}'.format(self.order))

        self.rules = self.named_outputs.get(
            'rules output', 'rules_{}'.format(self.order))

        self.attribute = parameters.get(self.ATTRIBUTE_PARAM)

        self.has_code = len(self.named_inputs) == 1

        self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = """
            from pyspark.mllib.fpm import FPGrowth
            from pyspark.sql.types import StructType, StructField, StringType, \
                FloatType, ArrayType

            def lift(sup_X_u_Y, n, sup_X, sup_Y):
                total = float(n)
                return (sup_X_u_Y / total) / (sup_X / total * sup_Y / total)
            def leverage():
                pass
            def conviction():
                pass

            # Current version of Spark supports FP-Growth only in RDD.
            # Assume that data is a line with transaction items separated by
            # space.
            data = {input}.rdd
            inx = reduce(
                lambda x, y: max(x, y),
                [inx for inx, attr in enumerate({input}.schema)
                    if attr.name == '{attr}'], 0)

            transactions = data.map(lambda line: line[inx])
            model = FPGrowth.train(
                transactions, minSupport={support}, numPartitions=10)

            emit_event(name='update task', message='Model trained',
                       status='RUNNING', identifier='{task_id}')

            items = model.freqItemsets()

            rules_schema = StructType(
                 [StructField("antecedent", StringType()),
                 [StructField("consequent", StringType()),
                 [StructField("confidence", FloatType())])

            gen_rules = []
            if items.isEmpty():
                schema = StructType(
                    [StructField("freq_item_sets", StringType()),
                    StructField("support", FloatType())])
                {output} = spark_session.createDataFrame([], schema)
                {output} = {output}.withColumn(
                    'support', {output}['freq'] / count)
            else:
                {output} = items.toDF(['freq_item_sets'])
                for rule in list(model._java_model.generateAssociationRules(
                    {confidence}):
                    gen_rules.append([
                        list(rule.antecedent()),
                        list(rule.consequent())
                        list(rule.confidence())
                    ])
            {rules} = spark_session.createDataFrame(gen_rules, rules_schema)
        """.format(input=self.named_inputs['input data'],
                   support=self.min_support,
                   output=self.output,
                   attr=self.attribute,
                   task_id=self.parameters['task']['id'],
                   rules=self.rules,
                   confidence=self.confidence)

        return dedent(code)
