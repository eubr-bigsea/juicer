from textwrap import dedent
from juicer.operation import Operation

#-------------------------------------------------------------------------------#
#
#                             Clustering Operations
#
#-------------------------------------------------------------------------------#


class ClusteringModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)


        if 'features' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}") \
                    .format('features', self.__class__))

        self.features = parameters['features'][0]
        self.model = self.named_outputs.get('model',
                                            'model_tmp_{}'.format(self.output))


        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")\
                    .format('train input data',  'algorithm', self.__class__))

        self.perform_transformation =  'output data' in self.named_outputs
        if not self.perform_transformation:
            self.output  = 'task_{}'.format(self.order)
        else:
            self.output = self.named_outputs['output data']
            self.prediction = self.parameters.get('prediction','prediction')

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):

        code = """
            cluster_model, settings = {algorithm}
            settings['features'] = '{features}'
            model = cluster_model.fit({input}, settings, numFrag)
            {model} = [cluster_model, model]
            """.format( model   =  self.model,
                        input   =  self.named_inputs['train input data'],
                        features    = self.features,
                        algorithm   = self.named_inputs['algorithm'])

        if self.perform_transformation:
            code += """
            settings['predCol'] = '{predCol}'
            {output} = cluster_model.transform({input}, model, settings, numFrag)
            """.format(output  = self.output,
                    input   = self.named_inputs['train input data'],
                    model   = self.model,
                    predCol = self.predCol)
        else:
            code += """
            {output} = None
            """.format(output  = self.output)

        return dedent(code)





class KMeansClusteringOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        attributes = ['number_of_clusters', 'max_iterations', 'init_mode'  ]
        for att in attributes:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}") \
                        .format(att, self.__class__))

        self.K = parameters['number_of_clusters']
        self.maxIters  = parameters['max_iterations']
        self.init_mode = parameters.get('init_mode','k-means||')
        self.epsilon = parameters.get('tolerance', 0.001)

        self.has_code = len(named_outputs) > 1

        if self.has_code:
            self.has_import = "from functions.ml.clustering.Kmeans.Kmeans " \
                              "import Kmeans\n"
            self.output = named_outputs.get('algorithm')


    def generate_code(self):
        code = """
            cluster_model = Kmeans()
            settings = dict()
            settings['k'] = {k}
            settings['maxIterations'] = {it}
            settings['epsilon']  = {ep}
            settings['initMode'] = '{init}'
            {output} = [cluster_model, settings]
            """.format( k       = self.K,
                        it      = self.maxIters,
                        ep      = self.epsilon,
                        init    = self.init_mode,
                        output  = self.output)
        return dedent(code)



#-------------------------------------------------------------------------------#
#
#                              Regression Operations
#
#-------------------------------------------------------------------------------#


class RegressionModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        if 'label' not in parameters and 'features' not in parameters:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}") \
                    .format('label',  'features', self.__class__))

        self.label = parameters['label'][0]
        self.features = parameters['features'][0]
        self.predCol  = parameters.get('prediction','prediction')
        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}") \
                    .format('train input data',  'algorithm', self.__class__))

        self.model = self.named_outputs.get('model',
                                            'model_tmp{}'.format(self.order))

        self.perform_transformation =  'output data' in self.named_outputs
        if not self.perform_transformation:
            self.output  = 'task_{}'.format(self.order)
        else:
            self.output = self.named_outputs['output data']
            self.prediction = self.parameters.get('prediction','prediction')

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])


    def generate_code(self):

        code = """
            numFrag = 4
            regressor, settings = {algorithm}
            settings['label'] = '{label}'
            settings['features'] = '{features}'
            model = regressor.fit({input}, settings, numFrag)
            {model} = [regressor, model]
            """.format( output    = self.output,
                        model     = self.model,
                        input     = self.named_inputs['train input data'],
                        algorithm = self.named_inputs['algorithm'],
                        label     = self.label,
                        features  = self.features)
        if self.perform_transformation:
            code += """
            settings['predCol'] = '{predCol}'
            {output} = regressor.transform({input}, model, settings, numFrag)
            """.format(predCol = self.predCol,
                        output  = self.output,
                        input   = self.named_inputs['train input data'])
        else:
            code += 'task_{} = None'.format(self.order)


        return dedent(code)




class LinearRegressionOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))
        self.maxIters = parameters.get('max_iter', 100)
        self.alpha = parameters.get('alpha', 0.001)
        self.has_import = "from functions.ml.regression.linearRegression." \
                          "linearRegression import linearRegression\n"

    def generate_code(self):
        code = """
            regression_model = linearRegression()
            settings = dict()
            settings['alpha']    = {alpha}
            settings['max_iter'] = {it}
            settings['option']   = 'SDG'
            {output} = [regression_model, settings]
            """.format( alpha   = self.alpha,
                        it      = self.maxIters,
                        output  = self.output)
        return dedent(code)
