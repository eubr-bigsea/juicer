# -*- coding: utf-8 -*-

from juicer.operation import Operation

class AndOperation(Operation): 
    """
    AND Operation; 
    No parameters;
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = (len(self.named_inputs)==4 and len(self.named_outputs) > 0)
        self.output = self.named_outputs.get('port_1y',
                                             'out_{}'.format(self.order)) 
        self.input_1a = self.named_inputs.get('port_1a')
        self.input_1b = self.named_inputs.get('port_1b')
        self.input_vcc = self.named_inputs.get('vcc_port')
        self.input_gnd = self.named_inputs.get('gnd_port')
        #import pdb; pdb.set_trace()

    def generate_code(self): 
        return f"{self.output} = {self.input_vcc} and {self.input_1a} and {self.input_1b}"  

