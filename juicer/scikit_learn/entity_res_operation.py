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
        if self.named_inputs.get('input data 1') is not None:
            self.input += self.named_inputs.get('input data 1')
        if self.named_inputs.get('input data 2') is not None:
            if len(self.input) > 0:
                self.input += ","
            self.input += self.named_inputs.get('input data 2')

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
            second_input = self.named_inputs.get('input data 2')
            third_input = self.named_inputs.get('input data 3')

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

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.ClassificationOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.input = ""
            self.intercept = float(parameters.get(self.INTERCEPT_PARAM, None))
            self.coefficients = parameters.get(self.COEFFICIENTS_PARAM, None)

            self.transpiler_utils.add_import("import recordlinkage as rl")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if self.named_inputs.get('input data') is not None:
            self.input = self.named_inputs.get('input data')

    def generate_code(self):
        if self.has_code:
            code = """
            logreg = rl.LogisticRegressionClassifier(coefficients=[{coefficients}], intercept={intercept})
            links = logreg.predict({input})
            {out} = links.to_frame().reset_index(drop=True)
            """.format(out=self.output,
                       input=self.input,
                       intercept=self.intercept,
                       coefficients=self.coefficients)
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
    """

    MATRIX_PARAM = 'confusion_matrix'
    F_SCORE_PARAM = 'f_score'
    RECALL_PARAM = 'recall'
    PRECISION_PARAM = 'precision'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.EvaluationOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.input = self.named_inputs.get('input data')
            self.indexing = self.named_inputs.get('indexing data')
            self.classification = self.named_inputs.get('classification data')

            self.confusion_matrix = int(parameters.get(self.MATRIX_PARAM, 1))
            self.f_score = int(parameters.get(self.F_SCORE_PARAM, 1))
            self.recall = int(parameters.get(self.RECALL_PARAM, 1))
            self.precision = int(parameters.get(self.PRECISION_PARAM, 1))

            self.transpiler_utils.add_import("import recordlinkage as rl")

            self.treatment()

    def treatment(self):
        if self.input is None or self.indexing is None or self.classification is None:
            #ERRO

    def generate_code(self):
        if self.has_code:
            code = """
            if {confusion_matrix} == 1:
                conf_logreg = rl.confusion_matrix({true_links}, {links}, len({candidate_links}))
            if {f_score} == 1:
                fscore = rl.fscore(conf_logreg)
            if {recall} == 1:
                recall = rl.recall({true_links}, {links})
            if {precision} == 1:
                precision = rl.precision({true_links}, {links})
            """.format(true_links=self.input,
                       candidate_links=self.indexing,
                       links=self.classification,
                       confusion_matrix=self.confusion_matrix,
                       f_score=self.f_score,
                       recall=self.recall,
                       precision=self.precision)
            return dedent(code)