from textwrap import dedent
from juicer.operation import Operation


class ClusteringModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_outputs) > 0 and len(self.named_inputs) == 2

        if self.FEATURES_ATTRIBUTE_PARAM not in parameters:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format(self.FEATURES_ATTRIBUTE_PARAM, self.__class__))

        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)[0]


        self.algorithm_cluster = "kmeans"
        self.output = self.named_outputs['output data']
        self.model = self.named_outputs.get('model', '{}'.format(self.output))
        # self.named_outputs['output data']))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'], self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.named_outputs['output data'], self.model])

    def generate_code(self):

        code = """
            cluster_model, model = {algorithm}
            numFrag = 4
            {output} = cluster_model.transform({input}, model, numFrag)
        """.format( output=self.output, model=self.model,
                    input=self.named_inputs['train input data'],
                    algorithm=self.named_inputs['algorithm'])


        return dedent(code)



class KMeansClusteringOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.Cluster_settings = {}
        self.Cluster_settings['k'] = parameters['number_of_clusters']
        self.Cluster_settings['maxIterations'] = parameters['max_iterations']
        self.Cluster_settings['epsilon'] = parameters['tolerance']
        self.Cluster_settings['initMode'] = 'kmeans++' #parameters['init_mode']


    def generate_code(self):
        code = """
            cluster_model = Kmeans()
            model         = cluster_model.fit({})
            {} = [custer_model, model]
            """.format(self.Cluster_settings,self.output)
        return dedent(code)

#---------------------------------------------------


class ClassificationModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_outputs) > 0 and len(self.named_inputs) == 2 ############


    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):

        if self.has_code:
            code = """
            ClassificationModel, settings = {algorithm}
            numFrag = 4
            {input} = Partitionize({input},numFrag)
            model = ClassificationModel.fit({input},settings,numFrag)

            {output} =  [ClassificationModel,model]
            """.format(output=self.output, input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'])

            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))

class ApplyModel(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2

    def generate_code(self):

        if self.has_code:
            code = """
            numFrag = 4
            {input} = Partitionize({input},numFrag)
            {output} = ClassificationModel.transform({input}, {model}, numFrag)
            """.format(input=self.named_inputs['input data'],
                       output=self.named_outputs['output data'],
                       model = self.named_inputs['model'])
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(self.named_inputs, self.__class__))

        return dedent(code)


class SvmClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)


        self.parameters = parameters
        self.has_code = True
        self.name = 'classification.SVM'
        self.coef_lambda        = parameters['coef_lambda']
        self.coef_lr            = parameters['coef_lr']
        self.coef_threshold     = parameters['coef_threshold']
        self.coef_maxIters      = parameters['coef_maxIters']


    def generate_code(self):
        code = """
            numFrag = 4
            ClassificationModel = SVM()
            settings = dict()
            settings['coef_lambda']    = {coef_lambda}
            settings['coef_lr']        = {coef_lr}
            settings['coef_threshold'] = {coef_threshold}
            settings['coef_maxIters']  = {coef_maxIters}

            {output} = [ClassificationModel, settings]
            """.format(coef_lambda = self.coef_lambda,
                       coef_lr     = self.coef_lr,
                       coef_threshold = self.coef_threshold,
                       coef_maxIters = self.coef_maxIters,
                       output=self.output)
        return dedent(code)