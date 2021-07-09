# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
from gettext import gettext
import pandas as pd


class IndexingOperation(Operation):
    """
    The indexing module is used to make pairs of records.
    These pairs are called candidate links or candidate matches.
    There are several indexing algorithms available such as blocking and sorted neighborhood indexing.
    Parameters:
    - expression
    """

    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.IndexingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            from juicer.scikit_learn.expression_er import ExpressionER
            self.expression = ''
            for json_code in self.parameters[self.EXPRESSION_PARAM]:
                expression = ExpressionER(json_code)
                self.expression += "\n" + expression.parsed_expression

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            self.transpiler_utils.add_import("from recordlinkage.index import *")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if len(self.expression) <= 1:
            raise ValueError(
                _("Parameter '{}' must be x>=2 for task {}").format(
                    self.EXPRESSION_PARAM, self.__class__))
        if self.named_inputs.get('indexing data') is not None:
            self.input = self.named_inputs.get('indexing data')

    def generate_code(self):
        if self.has_code:


            code = [
                "indexer = rl.Index()",
                code_columns,
                "{out} = indexer.index({input}).to_frame().reset_index(drop=True)"\
                ".rename({{0:'Record_1', 1:'Record_2'}}, axis=1)"
                .format(out=self.output, input=self.input)
            ]
            code = dedent("\n".join(code))
            return code

class ComparingOperation(Operation):
    """
    A set of informative, discriminating and independent features
     is important for a good classification of record pairs into
     matching and distinct pairs.
    The recordlinkage.Compare class and its methods can be used to
     compare records pairs. Several comparison methods are included
     such as string similarity measures, numerical measures and distance measures.
    Parameters:
    - expression
    """

    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.ComparingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            from juicer.scikit_learn.expression_er import ExpressionER
            self.expression = ''
            for json_code in self.parameters[self.EXPRESSION_PARAM]:
                expression = ExpressionER(json_code)
                self.expression += "\n" + expression.parsed_expression

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            self.transpiler_utils.add_import("from recordlinkage.compare import *")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if len(self.expression) <= 1:
            raise ValueError(
                _("Parameter '{}' must be x>=2 for task {}").format(
                        self.EXPRESSION_PARAM, self.__class__))
        if self.named_inputs.get('indexing data') is not None:
            self.input = self.named_inputs.get('indexing data')

    def generate_code(self):
        if self.has_code:
            code_compute = "{out} = compare.compute({input}".format(out=self.output, input=self.input)
            second_input = self.named_inputs.get('input data 3')
            third_input = self.named_inputs.get('input data 2')

            if second_input is not None:
                code_compute += "," + second_input
            if third_input is not None:
                code_compute += ","+ third_input
            code_compute +=")"

            code = [
                "compare = rl.Compare()",
                self.expression,
                "{input} = pd.MultiIndex.from_frame({input}, names=('Record_1', 'Record_2'))".format(input=self.input),
                code_compute
            ]
            code = dedent("\n".join(code))
            return code

class ClassificationOperation(Operation):
    """
    Classification is the step in the record linkage process
     were record pairs are classified into matches, non-matches
     and possible matches. Classification algorithms can be
     supervised or unsupervised (with or without training data).
    Parameters:
    - intercept: the interception value.
    - coefficients: the coefficients of the logistic regression.
    """

    INTERCEPT_PARAM = 'intercept'
    COEFFICIENTS_PARAM = 'coefficients'
    ALGORITHM_PARAM = 'algorithm'
    BINARIZE_PARAM = 'binarize'
    ALPHA_PARAM = 'alpha'
    USE_COL_NAMES_PARAM = 'use_col_names'


    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.ClassificationOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.input = ""
            self.comparing = ""
            self.intercept = parameters.get(self.INTERCEPT_PARAM, None)
            self.coefficients = parameters.get(self.COEFFICIENTS_PARAM, None)
            self.algorithm = parameters.get(self.ALGORITHM_PARAM, None)
            self.binarize = parameters.get(self.BINARIZE_PARAM, None)
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 0.0001))
            self.use_col_names = int(parameters.get(self.USE_COL_NAMES_PARAM, 1))

            self.transpiler_utils.add_import("import recordlinkage as rl")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if self.named_inputs.get('input data') is not None:
            self.input = self.named_inputs.get('input data')
        if self.named_inputs.get('comparing data') is not None:
            self.comparing = self.named_inputs.get('comparing data')
        if self.binarize is not None:
            self.binarize = float(self.binarize)
        if self.intercept is not None:
            self.intercept = float(self.intercept)
        if self.use_col_names == 1:
            self.use_col_names = True
        else:
            self.use_col_names = False

    def generate_code(self):
        if self.has_code:
            code = """
            if "{algorithm}" == "logistic-regression":
                classification = rl.LogisticRegressionClassifier(coefficients=[{coefficients}], intercept={intercept})
            elif "{algorithm}" == "svm":
                classification = rl.SVMClassifier()
            else:
                classification = rl.NaiveBayesClassifier(binarize={binarize}, alpha={alpha}, use_col_names={use_col_names})
            if {true_links} is not None:
                {true_links} = pd.MultiIndex.from_frame({true_links}, names=('Record_1', 'Record_2'))
                classification.fit({comparing_data}, {true_links})
            links = classification.predict({comparing_data})
            {out} = links.to_frame().reset_index(drop=True).rename({{0:'Record_1', 1:'Record_2'}}, axis=1)
            """.format(out=self.output,
                       comparing_data=self.comparing,
                       intercept=self.intercept,
                       coefficients=self.coefficients,
                       algorithm=self.algorithm,
                       binarize=self.binarize,
                       alpha=self.alpha,
                       use_col_names=self.use_col_names,
                       true_links=self.input)
            return dedent(code)

class EvaluationOperation(Operation):
    """
    Evaluation of classifications plays an important role in record linkage.
    Express your classification quality in terms accuracy, recall and F-score
     based on true positives, false positives, true negatives and false negatives.
    Parameters:
    - confusion_matrix
    - f_score
    - recall
    - precision
    - accuracy
    - specificity
    """

    MATRIX_PARAM = 'confusion_matrix'
    F_SCORE_PARAM = 'f_score'
    RECALL_PARAM = 'recall'
    PRECISION_PARAM = 'precision'
    ACCURACY_PARAM = 'accuracy'
    SPECIFICITY_PARAM = 'specificity'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.EvaluationOperation'

        self.display_text = self.parameters['task']['forms'].get(
            'display_text', {'value': 1}).get('value', 1) in (1, '1')
        self.display_image = self.parameters['task']['forms'].get(
            'display_image', {'value': 1}).get('value', 1) in (1, '1')

        self.has_code = len(self.named_inputs) == 3 and any([self.display_image,self.display_image])
        if self.has_code:
            self.input = self.named_inputs.get('input data')
            self.indexing = self.named_inputs.get('indexing data')
            self.classification = self.named_inputs.get('classification data')

            self.confusion_matrix = int(parameters.get(self.MATRIX_PARAM, 1))
            self.f_score = int(parameters.get(self.F_SCORE_PARAM, 1))
            self.recall = int(parameters.get(self.RECALL_PARAM, 1))
            self.precision = int(parameters.get(self.PRECISION_PARAM, 1))
            self.accuracy = int(parameters.get(self.ACCURACY_PARAM, 1))
            self.specificity = int(parameters.get(self.SPECIFICITY_PARAM, 1))

            self.transpiler_utils.add_import("import recordlinkage as rl")

            self.treatment()

    def treatment(self):
        if any([self.input is None, self.indexing is None, self.classification is None]):
            msg = _("Parameters '{}', '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                'input data', 'indexing data', 'classification data', self.__class__.__name__))

    def generate_code(self):
        if self.has_code:
            code = """
            metrics = []
            display_text = {display_text}
            display_image = {display_image}
            
            {true_links} = pd.MultiIndex.from_frame({true_links})
            {links} = pd.MultiIndex.from_frame({links})
            
            conf_logreg = rl.confusion_matrix({true_links}, {links}, len({candidate_links}))
            if {f_score} == 1:
                fscore = rl.fscore(conf_logreg)
                metrics.append(['F-Score',fscore])
            if {recall} == 1:
                recall = rl.recall({true_links}, {links})
                metrics.append(['Recall',recall])
            if {precision} == 1:
                precision = rl.precision({true_links}, {links})
                metrics.append(['Precision',precision])
            if {accuracy} == 1:
                accuracy = rl.accuracy({true_links}, {links}, len({candidate_links}))
                metrics.append(['Accuracy',accuracy])
            if {specificity} == 1:
                specificity = rl.specificity({true_links}, {links}, len({candidate_links}))
                metrics.append(['Specificity',specificity])
            if display_text:
                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        {table_headers}, metrics, title='{title}').generate()
                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content,
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
            if display_image and {confusion_matrix}:
                content = ConfusionMatrixImageReport(
                    cm=conf_logreg,classes=[0,1]).generate(submission_lock)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content,
                    type='IMAGE', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
            """.format(true_links=self.input,
                       candidate_links=self.indexing,
                       links=self.classification,
                       confusion_matrix=self.confusion_matrix,
                       f_score=self.f_score,
                       recall=self.recall,
                       precision=self.precision,
                       display_text=self.display_text,
                       display_image=self.display_image,
                       title=_('Evaluation result'),
                       table_headers=[_('Metric'), _('Value')],
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       specificity=self.specificity,
                       accuracy=self.accuracy)
            return dedent(code)