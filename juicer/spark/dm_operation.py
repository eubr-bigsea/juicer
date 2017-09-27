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
            raise ValueError(_(
                'Support must be informed for classifier {}').format(
                self.__class__))

        self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))
        if self.min_support < .1:
            raise ValueError('Support must be greater or equal to 0.1')

        self.output = self.named_outputs.get(
            'output data', 'freq_items_{}'.format(self.order))

        self.rules = self.named_outputs.get(
            'rules output', 'rules_{}'.format(self.order))

        self.attribute = parameters.get(self.ATTRIBUTE_PARAM)
        if not self.attribute:
            raise ValueError(
                'Missing parameter {}'.format(self.ATTRIBUTE_PARAM))
        self.attribute = self.attribute[0]

        self.has_code = len(self.named_inputs) == 1

        self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.rules])

    def get_data_out_names(self, sep=','):
        return self.get_output_names(sep)

    def generate_code(self):
        code = u"""
            try:
                from pyspark.ml.fpm import FPGrowth
                algorithm = FPGrowth(itemsCol="{attr}",
                    minSupport={support}, minConfidence=0.6) # FIXME

                # Evaluate if using cache is a good idea
                {input}.cache()
                size = float({input}.count())
                model = algorithm.fit({input})

                emit_event(name='update task', message='{model_trained}',
                           status='RUNNING', identifier='{task_id}')
                {output} = model.freqItemsets.withColumn('relative_support',
                    functions.col('freq') / size)
                {rules} = model.associationRules
            except Exception as e:
                sparkException = 'org.apache.spark.SparkException'
                if hasattr(e, 'java_exception') and \
                        e.java_exception.getClass().getName() == sparkException:
                    cause = e.java_exception.getCause()
                    not_unique = 'Items in a transaction must be unique'
                    if cause and cause.getMessage().find(not_unique) > -1:
                        raise ValueError('{not_unique}')
                    else:
                        raise
                else:
                    raise
        """.format(input=self.named_inputs['input data'],
                   support=self.min_support,
                   output=self.output,
                   attr=self.attribute,
                   task_id=self.parameters['task']['id'],
                   rules=self.rules,
                   not_unique=_('Items in a transaction must be unique'),
                   confidence=self.confidence,
                   model_trained=_('Model trained'))

        return dedent(code)


class AssociationRulesOperation(Operation):
    """
    AssociationRules constructs rules that have a single item as the consequent.
    Current implementation uses FP-Growth.
    """
    CONFIDENCE_PARAM = 'confidence'
    RULES_COUNT_PARAM = 'rules_count'
    ATTRIBUTE_PARAM = 'attribute'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))
            self.rules_count = parameters.get(self.RULES_COUNT_PARAM, 200)
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM)

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = """
            from pyspark.mllib.fpm import FPGrowth
            from pyspark.sql.types import StructType, StructField, StringType
            # Current version of Spark (2.1) supports FP-Growth only in RDD.
            # Assume that data is a line with transaction items separated by
            # space.
            # TODO: Use correct class
            data = {input}.rdd
            # data.cache()
            #
            # inx = reduce(
            #     lambda a, b: max(a, b),
            #     [inx for inx, x in enumerate({input}.schema)
            #         if x.name == '{attr}'], 0)
            #
            # transactions = data.map(lambda line:
            #     [int(v) for v in line[inx].strip().split(' ')])
            model = FPGrowth.train(
                data, minSupport={support}, numPartitions=10)

            # Experimental code!
            rules = sorted(
               model._java_model.generateAssociationRules(0.9).collect(),
               key=lambda x: x.confidence(), reverse=True)
            for rule in rules[:{rules_count}]:
               print rule

            emit_event(name='update task', message='{model_trained}',
                       status=_('RUNNING'), identifier='{task_id}')

            items = model.freqItemsets()
            if items.isEmpty():
                schema = StructType(
                    [StructField("freq_item_sets", StringType(), True), ])
                {output} = spark_session.createDataFrame([], schema)
            else:
                {output} = items.toDF(['freq_item_sets'])
        """.format(input=self.named_inputs['input data'],
                   confidence=self.confidence,
                   output=self.output,
                   rules_count=self.rules_count,
                   attr=self.attribute[0],
                   task_id=self.parameters['task']['id'],
                   model_trained=_('Model trained'))

        return dedent(code)


class SequenceMiningOperation(Operation):
    """
    Sequential pattern mining algorithm (PrefixSpan)
    """
    MIN_SUPPORT_PARAM = 'min_support'
    MAX_PATTERN_LENGTH_PARAM = 'max_pattern_length'
    ATTRIBUTE_PARAM = 'attribute'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.min_support = float(
                parameters.get(self.MIN_SUPPORT_PARAM, 0.1))
            self.rules_count = parameters.get(self.MAX_PATTERN_LENGTH_PARAM, 10)
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM)

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = """
            from pyspark.mllib.fpm import PrefixSpan
            from pyspark.sql.types import StructType, StructField, StringType
            # Current version of Spark (2.1) supports only RDD.

            data = {input}.rdd
            data.cache()

            inx = reduce(
                lambda a, b: max(a, b),
                [inx for inx, x in enumerate({input}.schema)
                    if x.name == '{attr}'], 0)

            transactions = data.map(lambda line:
                [int(v) for v in line[inx].strip().split(' ')])
            model = FPGrowth.train(
                transactions, minSupport={support}, numPartitions=10)

            # Experimental code!
            rules = sorted(
               model._java_model.generateAssociationRules(0.9).collect(),
               key=lambda x: x.confidence(), reverse=True)
            for rule in rules[:{rules_count}]:
               print rule

            emit_event(name='update task', message=_('Model trained'),
                       status=_('RUNNING'), identifier='{task_id}')

            items = model.freqItemsets()
            if items.isEmpty():
                schema = StructType(
                    [StructField("freq_item_sets", StringType(), True), ])
                {output} = spark_session.createDataFrame([], schema)
            else:
                {output} = items.toDF(['freq_item_sets'])
        """.format(input=self.named_inputs['input data'],
                   confidence=self.min_support,
                   output=self.output,
                   rules_count=self.rules_count,
                   attr=self.attribute[0],
                   task_id=self.parameters['task']['id'])
