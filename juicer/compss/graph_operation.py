# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class PageRankOperation(Operation):
    """
        Run PageRank for a fixed number of iterations returning a
        graph with vertex attributes containing the PageRank.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)
        self.has_code   = len(named_inputs) == 1 and\
                          len(named_outputs)>0 and\
                          len(self.parameters['inlink']) > 0 and\
                          len(self.parameters['outlink']) > 0

        if self.has_code:
            self.has_import = "from functions.graph.PageRank.pagerank import *\n"
            self.generate_code()

    def generate_code(self):

        code = """

        numFrag = 4
        settings = dict()
        settings['inlink']   = '{inlink}'
        settings['outlink']  = '{outlink}'
        settings['maxIters'] = {iter}
        settings['damping_factor'] = {damping_factor}

        pr = PageRank()
        {out} = pr.runPageRank({input}, settings, numFrag)

        """.format( out     =   self.named_outputs['output data'],
                    input   =   self.named_inputs['input data'],
                    inlink  =   self.parameters['inlink'][0],
                    outlink =   self.parameters['outlink'][0],
                    iter    =   self.parameters['maxIters'],
                    damping_factor = self.parameters['damping_factor'])
        return dedent(code)