# -*- coding: utf-8 -*-



from textwrap import dedent

from juicer.operation import Operation


class EntityMatchingOperation(Operation):
    ALGORITHM_BULMA = 'BULMA'

    ALGORITHM_PARAM = 'algorithm'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ALGORITHM_PARAM in parameters:
            self.algorithm = parameters[self.ALGORITHM_PARAM]
        else:
            raise ValueError(_(
                "Parameter '{}' must be informed for task {}").format(
                self.ALGORITHM_PARAM, self.__class__))

        if self.algorithm not in [self.ALGORITHM_BULMA]:
            raise ValueError(_(
                "Algorithm '{}' not supported in for task {}").format(
                self.algorithm, self.__class__))
        self.output = self.named_outputs.get('output data 1',
                                             'em_data_{}'.format(
                                                 self.order))
        self.has_code = len(self.named_inputs) == 2

    def generate_code(self):
        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        if self.algorithm == self.ALGORITHM_BULMA:
            code = """
                # noinspection PyProtectedMember
                from pyspark.sql.dataframe import DataFrame
                jvm = spark_session.sparkContext._gateway.jvm
                try:
                    matching_routes = jvm.LineMatching20.MatchingRoutesV2()

                    mapped_dataset = matching_routes.generateDataFrames(
                        {df_shape}._jdf, {df_gps}._jdf, 1,
                        spark_session._jsparkSession)

                    result = matching_routes.run(mapped_dataset, 1,
                        spark_session._jsparkSession)
                    {out} = DataFrame(result.toDF(), spark_session)
                except TypeError as te:
                    raise TypeError('Unable to load BULMA classes. Please '
                        'check CLASSPATH and libraies')
            """.format(df_shape=input_data1, df_gps=input_data2,
                       out=self.output)
        else:
            code = ""
        return dedent(code)
