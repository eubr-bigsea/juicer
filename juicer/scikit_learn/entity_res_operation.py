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
    - attributes: list of the attributes that will be used to do de indexing.
    - alg: the algorithm that will ne used for the indexing.
    """

    ATTRIBUTES_PARAM = 'attributes'
    ALGORITHM_PARAM = 'alg'
    WINDOW_PARAM = 'window'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.IndexingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.attributes = parameters['attributes']
            self.alg = parameters.get(self.ALGORITHM_PARAM, "Block")
            self.window = int(parameters.get(self.WINDOW_PARAM, 3))

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            if self.alg == "Block":
                self.transpiler_utils.add_import("from recordlinkage.index import Block")
            elif self.alg == "Full":
                self.transpiler_utils.add_import("from recordlinkage.index import Full")
            elif self.alg == "Random":
                self.transpiler_utils.add_import("from recordlinkage.index import Random")
            elif self.alg == "Sorted Neighbourhood":
                self.transpiler_utils.add_import("from recordlinkage.index import SortedNeighbourhood")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if len(self.attributes) == 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        if self.named_inputs.get('input data 2') is not None:
            self.input += self.named_inputs.get('input data 2')
        if self.named_inputs.get('input data 1') is not None:
            if len(self.input) > 0:
                self.input += ","
            self.input += self.named_inputs.get('input data 1')

    def generate_code(self):
        if self.has_code:
            code_columns = None
            if self.alg == "Sorted Neighbourhood":
                code_columns = "\n".join(["indexer.sortedneighbourhood('{col}', window={window})"
                                         .format(col=col,window=self.window) for col in self.attributes])
            elif self.alg == "Block":
                code_columns = "\n".join(["indexer.block('{col}')".format(col=col) for col in self.attributes])
            elif self.alg == "Random":
                code_columns = "\n".join(["indexer.random('{col}')".format(col=col) for col in self.attributes])
            elif self.alg == "Full":
                code_columns = "\n".join(["indexer.full('{col}')".format(col=col) for col in self.attributes])

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
    - attributes: list of the attributes that will be used to do de comparing.
    """

    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.ComparingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.attributes = parameters['attributes']

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            self.transpiler_utils.add_import("from recordlinkage.compare import Exact")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if len(self.attributes) <= 1:
            raise ValueError(
                _("Parameter '{}' must be x>=2 for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        if self.named_inputs.get('indexing data') is not None:
            self.input = self.named_inputs.get('indexing data')

    def generate_code(self):
        if self.has_code:
            code_columns = "\n".join(["compare.exact('{col}','{col}', label='{col}')".format(col=col) for col in self.attributes])
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
                code_columns,
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
            self.intercept = float(parameters.get(self.INTERCEPT_PARAM, None))
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
        if self.use_col_names == 1:
            self.use_col_names = True
        else:
            self.use_col_names = False

    def generate_code(self):
        if self.has_code:
            code = """
            if "{algorithm}" == "logistic-regression":
                class = rl.LogisticRegressionClassifier(coefficients=[{coefficients}], intercept={intercept})
            elif "{algorithm}" == "svm":
                class = rl.SVMClassifier()
                class.fit({input}, {true_links})
            else:
                class = rl.NaiveBayesClassifier(binarize={binarize}, alpha={alpha}, use_col_names={use_col_names})
                class.fit({input}, {true_links})
            links = class.predict({input})
            {out} = links.to_frame().reset_index(drop=True).rename({{0:'Record_1', 1:'Record_2'}}, axis=1)
            """.format(out=self.output,
                       input=self.comparing,
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