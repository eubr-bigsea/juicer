from textwrap import dedent
from juicer.operation import Operation


#-------------------------------------------------------------------------------#
#
#                                 Model Operations
#
#-------------------------------------------------------------------------------#

class ApplyModel(Operation):

    """
    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'features' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}") \
                    .format('features', self.__class__))

        self.prediction = parameters.get('prediction','prediction')

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(self.order))

        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                    .format('input data',  'model', self.__class__))


    def generate_code(self):

        code = """
        algorithm, model = {model}
        settings = dict()
        settings['features'] = '{features}'
        settings['predCol']  = '{predCol}'

        {output} = algorithm.transform({input}, model, settings, numFrag)
        """.format( input    = self.named_inputs['input data'],
                    model    = self.named_inputs['model'],
                    features = self.parameters['features'][0],
                    predCol  = self.prediction,
                    output   = self.output)
        return dedent(code)


class EvaluateModelOperation(Operation):

    """
    REVIEW: 2017-10-20
    ??? - Juicer ?? / Tahiti ok/ implementation ok

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        for att in ['label_attribute','prediction_attribute', 'metric']:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}") \
                        .format(att, self.__class__))

        self.true_col = self.parameters['label_attribute'][0]
        self.pred_col = self.parameters['prediction_attribute'][0]
        self.metric = self.parameters['metric']

        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                    .format('input data',  'model', self.__class__))


        if self.metric in ['rmse','mse','mae']:
            self.modeltype  = 'RegressionModelEvaluation'
            self.has_import = \
                "from functions.ml.metrics." \
                "RegressionModelEvaluation import *\n"
        else:
            self.modeltype  = 'ClassificationModelEvaluation'
            self.has_import = \
                "from functions.ml.metrics.ClassificationModelEvaluation" \
                " import *\n"

        self.evaluated_out = \
            self.named_outputs.get('evaluated model',
                                   'evaluated_model{}'.format(self.order))
        self.evaluator = \
            self.named_outputs.get("evaluator", 'evaluator{}'.format(self.order))



    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.evaluated_out, self.evaluator])

    def generate_code(self):

        code = """
            settings = dict()
            settings['pred_col'] = '{pred_col}'
            settings['test_col'] = '{true_col}'
            settings['metric'] = '{metric}'
            {evaluator} = {type}()
            {evaluated_model} = {evaluator}.calculate({input}, settings, numFrag)

            """.format(input    = self.named_inputs['input data'],
                       type     = self.modeltype,
                       metric   = self.metric,
                       true_col = self.true_col,
                       pred_col = self.pred_col,
                       evaluated_model = self.evaluated_out,
                       evaluator = self.evaluator)

        return dedent(code)



class LoadModel(Operation):

    """
    REVIEW: 2017-10-20
    ??? - Juicer ?? / Tahiti ok/ implementation ok

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'name' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}") \
                    .format('name', self.__class__))


        self.filename = parameters['name']
        self.output = named_outputs.get('output data',
                                        'output_data_{}'.format(self.order))

        self.has_code  = len(named_outputs) > 0
        if self.has_code:
            self.has_import = 'from functions.ml.Models ' \
                              'import LoadModelFromHDFS\n'
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}") \
                    .format('output data', self.__class__))

    def generate_code(self):

        code = """
        settings = dict()
        settings['path'] = {filename}
        {model} = LoadModelFromHDFS(settings)
        """.format(model = self.output,
                   filename = self.filename)
        return dedent(code)


class SaveModel(Operation):

    """
    REVIEW: 2017-10-20
    ??? - Juicer ?? / Tahiti ok/ implementation ok

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code  = 'model' in parameters
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}") \
                    .format('model', self.__class__))
        self.filename  = named_inputs['model']
        self.overwrite = parameters.get('write_nome', 'OVERWRITE')
        if self.overwrite == 'OVERWRITE':
            self.overwrite = True
        else:
            self.overwrite = False

        if self.has_code:
            self.has_import = 'from functions.ml.Models ' \
                              'import SaveModelToHDFS\n'


    def generate_code(self):

        code = """
        settings = dict()
        settings['path'] = {filename}
        settings['overwrite'] = {overwrite}

        sucess = SaveModelToHDFS({input}, settings)
        """.format(input = self.named_inputs['model'],
                   filename = self.filename,
                   overwrite = self.overwrite)
        return dedent(code)