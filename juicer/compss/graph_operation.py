# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class PageRankOperation(Operation):
    """
    Run PageRank for a fixed number of iterations returning a
    graph with vertex attributes containing the PageRank.

    REVIEW: 2017-10-23
    OK - Juicer / Tahiti / implementation ?
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        if 'inlink' not in parameters and 'outlink' not in parameters:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                    .format('inlink',  'outlink', self.__class__))

        self.has_code   = len(named_inputs) == 1

        self.inlink = parameters['inlink'][0]
        self.outlink = parameters['outlink'][0]
        self.maxIters = parameters.get('maxIters', 100)
        self.damping_factor = parameters.get('damping_factor', 0.85)
        self.output = named_outputs.get('output data',
                                        'output_data_{}'.format(self.order))

        if self.has_code:
            self.has_import = "from functions.graph.PageRank.pagerank " \
                              "import PageRank\n"


    def generate_code(self):

        code = """
        settings = dict()
        settings['inlink']   = '{inlink}'
        settings['outlink']  = '{outlink}'
        settings['maxIters'] = {iter}
        settings['damping_factor'] = {damping_factor}

        pr = PageRank()
        {out} = pr.runPageRank({input}, settings, numFrag)
        """.format( out     =   self.output,
                    input   =   self.named_inputs['input data'],
                    inlink  =   self.inlink,
                    outlink =   self.outlink,
                    iter    =   self.maxIters,
                    damping_factor = self.damping_factor)
        return dedent(code)