from textwrap import dedent
from juicer.operation import Operation


#----------------------------------------------------------------------------------------------------------------------#
#
#                                                   Associative Operations
#
#----------------------------------------------------------------------------------------------------------------------#

class AprioriOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code =     len(self.named_outputs) >= 1 and len(self.named_inputs) == 1

        self.output   = self.named_outputs.get('output data', 'tmp_items_{}'.format(self.order))
        self.perform_genRules =  'rules output' in self.named_outputs

        if not self.perform_genRules:
            self.named_outputs['rules output'] = 'rules_{}_tmp'.format(self.order)

        if self.has_code:
            self.has_import = "from functions.ml.associative.apriori.apriori import *\n"

    def generate_code(self):
        if self.has_code:
            code = """
                numFrag = 4
                settings = dict()
                settings['col'] = "{col}"
                settings['minSupport'] = {supp}

                apriori = Apriori()
                {output} = apriori.runApriori({input}, settings, numFrag)
                """.format( output = self.output,
                            input  = self.named_inputs['input data'],
                            col     = self.parameters['attribute'][0],
                            supp  = self.parameters['min_support'])

            if self.perform_genRules:
                code+="""
                settings['confidence']  = {conf}
                {rules} = apriori.generateRules({output},settings)
                """.format( output = self.output,
                            rules  = self.named_outputs['rules output'],
                            conf = self.parameters['confidence'])
            else:
                code+="""
                {rules} = None
                """.format( rules  = self.named_outputs['rules output'])
            return dedent(code)

        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))


class AssociationRulesOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code =     len(self.named_outputs) == 1 and len(self.named_inputs) == 1

        if self.has_code:
            self.has_import = "from functions.ml.associative.AssociationRules import AssociationRulesOperation\n"

    def generate_code(self):
        if self.has_code:
            code = """
                settings = dict()
                settings['col'] = "{col}"
                settings['rules_count'] = {total}
                settings['confidence']  = {conf}
                {output} = AssociationRulesOperation({input}, settings)
                """.format( output = self.output,
                            input  = self.named_inputs['input data'],
                            col    = self.parameters['attribute'][0],
                            conf   = self.parameters['confidence'],
                            total  = self.parameters['rules_count'])

            return dedent(code)

        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))





#----------------------------------------------------------------------------------------------------------------------#
#
#                                                   Feature Extraction Operations
#
#----------------------------------------------------------------------------------------------------------------------#

class FeatureAssemblerOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code =     len(self.named_outputs) == 1 \
                        and len(self.named_inputs) == 1 \
                        and 'attributes' in self.parameters\
                        and 'alias' in self.parameters
        if self.has_code:
            self.has_import = "from functions.ml.FeatureAssembler import FeatureAssemblerOperation\n"

    def generate_code(self):

        if self.has_code:
            code = """
                numFrag = 4
                columns = {columns}
                alias   = '{alias}'
                {output} = FeatureAssemblerOperation({input}, columns, alias, numFrag)
                """.format( output = self.named_outputs['output data'],
                            input  = self.named_inputs['input data'],
                            columns= self.parameters['attributes'],
                            alias  = self.parameters['alias'])

            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))


class FeatureIndexerOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code =     len(self.named_outputs) == 1 \
                            and len(self.named_inputs) == 1 \
                            and 'attributes' in self.parameters \
                            and 'alias' in self.parameters
        if self.has_code:
            self.has_import = "from functions.ml.FeatureAssembler import FeatureAssemblerOperation\n"

    def generate_code(self):
        if self.has_code:

            code = """
                numFrag = 4
                settings = dict()
                settings['inputCol'] = {columns}
                settings['outputCol'] = '{alias}'
                settings['IndexToString'] = False
                {output}, mapper = FeatureIndexerOperation({input}, settings, numFrag)
                """.format( output = self.named_outputs['output data'],
                            input  = self.named_inputs['input data'],
                            columns= self.parameters['attributes'],
                            alias  = self.parameters['alias'])

            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))


#----------------------------------------------------------------------------------------------------------------------#
#
#                                                   Model Operations
#
#----------------------------------------------------------------------------------------------------------------------#

class ApplyModel(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2 and len(self.named_outputs) == 1
        #Verificar se tem features

    def generate_code(self):

        if self.has_code:
            code = """
            numFrag  = 4
            algorithm, model = {model}
            settings = dict()
            settings['features']  = '{features}'
            settings['predCol'] = '{predCol}'

            {output}    = algorithm.transform({input}, model, settings, numFrag)
            """.format( input      = self.named_inputs['input data'],
                        model      = self.named_inputs['model'],
                        features   = self.parameters['features'][0],
                        predCol    = self.parameters.get('prediction','prediction'),
                        output     = self.named_outputs['output data'])
            return dedent(code)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(self.named_inputs, self.__class__))




#----------------------------------------------------------------------------------------------------------------------#
#
#                                                   Clustering Operations
#
#----------------------------------------------------------------------------------------------------------------------#


class ClusteringModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_outputs) > 0 and len(self.named_inputs) == 2
        self.model  = self.named_outputs.get('model', '{}_tmp'.format(self.output))
        self.perform_transformation =  'output data' in self.named_outputs
        if not self.perform_transformation:
            self.named_outputs['output data'] = 'task_{}'.format(self.order)

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'], self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.named_outputs['output data'],  self.model])

    def generate_code(self):

        if self.has_code:
            code = """
                cluster_model, settings = {algorithm}
                numFrag = 4
                settings['features'] = '{features}'
                model = cluster_model.fit({input}, settings, numFrag)
                {model} = [cluster_model, model]
                """.format( model   =    self.model,
                            input   =   self.named_inputs['train input data'],
                            features    =   self.parameters['features'][0],
                            algorithm   =   self.named_inputs['algorithm'])
            if self.perform_transformation:
                code += """
                settings['predCol'] = '{predCol}'
                {output} = cluster_model.transform({input}, model, settings, numFrag)"""\
                .format(output  = self.named_outputs['output data'],
                        input   = self.named_inputs['train input data'],
                        model   = self.model,
                        predCol = self.parameters['prediction'])
            return dedent(code)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(self.named_inputs, self.__class__))





class KMeansClusteringOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_import = "from functions.ml.clustering.Kmeans.Kmeans import *\n"

    def generate_code(self):
        code = """
            cluster_model = Kmeans()
            numFrag  = 4
            settings = dict()
            settings['k'] = {k}
            settings['maxIterations'] = {it}
            settings['epsilon'] = {ep}
            settings['initMode'] = '{init}'
            {output} = [cluster_model, settings]
            """.format( k       = self.parameters['number_of_clusters'],
                        it      = self.parameters['max_iterations'],
                        ep      = self.parameters['tolerance'],
                        init    = self.parameters['init_mode'],
                        output  = self.named_outputs['algorithm'])
        return dedent(code)

#----------------------------------------------------------------------------------------------------------------------#
#
#                                                   Classification Operations
#
#----------------------------------------------------------------------------------------------------------------------#


class ClassificationModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code =  len(self.named_outputs) > 0 and len(self.named_inputs) == 2
        self.model    = self.named_outputs.get('model', '{}_tmp'.format(self.output))
        self.perform_transformation =  'output data' in self.named_outputs
        if not self.perform_transformation:
            self.named_outputs['output data'] = 'task_{}'.format(self.order)

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.named_outputs['output data'], self.model])

    def generate_code(self):
        if self.has_code:
            code = """
                ClassificationModel, settings = {algorithm}
                numFrag = 4
                settings['label']     = '{label}'
                settings['features']  = '{features}'
                model   = ClassificationModel.fit({input}, settings, numFrag)
                {model} = [ClassificationModel, model]
                """.format( model      = self.model,
                            input      = self.named_inputs['train input data'],
                            algorithm  = self.named_inputs['algorithm'],
                            label      = self.parameters['label'][0],
                            features   = self.parameters['features'][0])

            if self.perform_transformation:
                code += """
                settings['predCol'] = '{predCol}'
                {output} = ClassificationModel.transform({input}, model, settings, numFrag)"""\
                    .format(predCol = self.parameters['prediction'],
                            output  = self.named_outputs['output data'],
                            input   = self.named_inputs['train input data'])
            else:
                code += 'task_{} = None'.format(self.order)

            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))



class LogisticRegressionOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.has_import = "from functions.ml.classification.LogisticRegression.logisticRegression import *\n"

    def generate_code(self):
        if self.has_code:

            code = """
                ClassificationModel = logisticRegression()
                settings = dict()
                settings['alpha']       = {alpha}
                settings['iters']       = {maxIters}
                settings['threshold']   = {threshold}
                settings['regularization'] = {regularization}
                {output} = [ClassificationModel, settings]
                """.format(alpha            = self.parameters['coef_alpha'],
                           regularization   = self.parameters['coef_lr'],
                           threshold        = self.parameters['coef_threshold'],
                           maxIters         = self.parameters['max_iter'],
                           output           = self.named_outputs['algorithm'])
            return dedent(code)
        else:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format( '[]outputs', self.__class__))




class SvmClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.has_import = "from functions.ml.classification.Svm.svm import *\n"

    def generate_code(self):
        if self.has_code:
            code = """
                ClassificationModel = SVM()
                settings = dict()
                settings['coef_lambda']    = {coef_lambda}
                settings['coef_lr']        = {coef_lr}
                settings['coef_threshold'] = {coef_threshold}
                settings['coef_maxIters']  = {coef_maxIters}

                {output} = [ClassificationModel, settings]
                """.format(coef_lambda      = self.parameters['coef_lambda'],
                           coef_lr          = self.parameters['coef_lr'],
                           coef_threshold   = self.parameters['coef_threshold'],
                           coef_maxIters    = self.parameters['max_iter'],
                           output           = self.named_outputs['algorithm'])
            return dedent(code)
        else:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format( '[]outputs', self.__class__))


class NaiveBayesClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.has_import = "from functions.ml.classification.NaiveBayes.naivebayes import *\n"

    def generate_code(self):
        if self.has_code:

            code = """
                ClassificationModel = GaussianNB()
                {output} = [ClassificationModel, dict()]
                """.format(output = self.named_outputs['algorithm'])
            return dedent(code)
        else:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format( '[]outputs', self.__class__))


class KNNClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.has_import = "from functions.ml.classification.Knn.knn import *\n"

    def generate_code(self):
        if self.has_code:

            code = """
                ClassificationModel = KNN()
                settings      = dict()
                settings['K'] = {K}
                {output} = [ClassificationModel, settings]
                """.format(K      = self.parameters['k'],
                           output = self.named_outputs['algorithm'])
            return dedent(code)
        else:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format( '[]outputs', self.__class__))

class EvaluateModelOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = (len(self.named_inputs) == 2)

        if self.has_code:
            self.metric = self.parameters['metric']
            if self.metric in ['rmse','mse','mae']:
                self.modeltype  = 'RegressionModelEvaluation'
                self.has_import = "from functions.ml.metrics.RegressionModelEvaluation import *\n"
            else:
                self.modeltype  = 'ClassificationModelEvaluation'
                self.has_import = "from functions.ml.metrics.ClassificationModelEvaluation import *\n"

            self.evaluated_out = self.named_outputs.get('evaluated model', 'evaluated_model{}'.format(self.order))
            self.evaluator     = self.named_outputs.get("evaluator", 'evaluator{}'.format(self.order))


    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.evaluated_out, self.evaluator])

    def generate_code(self):
        if self.has_code:

            code = """
                    numFrag  = 4
                    settings = dict()
                    settings['pred_col'] = {pred_col}
                    settings['test_col'] = {true_col}
                    settings['metric'] = '{metric}'
                    {evaluator}  = {type}()
                    {evaluated_model} = {evaluator}.calculate({input},settings,numFrag)

                    """.format(input    = self.named_inputs['input data'],
                               type     = self.modeltype,
                               metric   = self.metric,
                               true_col = self.parameters['label_attribute'][0],
                               pred_col = self.parameters['prediction_attribute'][0],
                               evaluated_model   = self.evaluated_out,
                               evaluator = self.evaluator)

            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))


#----------------------------------------------------------------------------------------------------------------------#
#
#                                                   Regression Operations
#
#----------------------------------------------------------------------------------------------------------------------#


class RegressionModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_outputs) > 0 and len(self.named_inputs) == 2
        self.model  = self.named_outputs.get('model', '{}_tmp'.format(self.output))
        self.perform_transformation =  'output data' in self.named_outputs
        if not self.perform_transformation:
            self.named_outputs['output data'] = 'task_{}'.format(self.order)

    def get_data_out_names(self, sep=','):
            return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.named_outputs['output data'], self.model])

    def generate_code(self):

        if self.has_code:
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
                            label     = self.parameters['label'][0],
                            features  = self.parameters['features'][0])
            if self.perform_transformation:
                code += """
                settings['predCol'] = '{predCol}'
                {output} = regressor.transform({input}, model, settings, numFrag)""" \
                    .format(predCol = self.parameters['prediction'],
                            output  = self.named_outputs['output data'],
                            input   = self.named_inputs['train input data'])
            else:
                code += 'task_{} = None'.format(self.order)


            return dedent(code)
        else:
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format('[]inputs', '[]outputs', self.__class__))



class LinearRegressionOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.has_import = "from functions.ml.regression.linearRegression.linearRegression import *\n"

    def generate_code(self):

        if self.has_code:

            code = """
                regression_model = linearRegression()
                settings = dict()
                settings['alpha']    = {alpha}
                settings['max_iter'] = {it}
                settings['option']   = 'SDG'
                {output} = [regression_model, settings]
                """.format( alpha   = self.parameters['alpha'],
                            it      = self.parameters['max_iter'],
                            output  = self.named_outputs['algorithm'])
            return dedent(code)
        else:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format( '[]outputs', self.__class__))