# -*- coding: utf-8 -*-

#import json
#import time
#from random import random
#from textwrap import dedent

from juicer.operation import Operation

class Fonte5vOperation(Operation): 
    """
    Check if source is turn on or turn off; 
    Parameters:
    - on_off: source turn on or turn off; 
    """
    ON_OFF_PARAM = 'voltage' 

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.voltage = parameters.get(self.ON_OFF_PARAM) in ('1', 1)
        self.voltage  = 1 if self.voltage else 0
        self.has_code = len(self.named_outputs) > 0 
        self.output = named_outputs.get('output_port',
                                             'out_{}'.format(self.order)) 

    def generate_code(self): 
        return f"{self.output} = {self.voltage}"

class TerraOperation(Operation): 
    """
    It receives zero.  
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = (len(self.named_outputs) > 0)
        self.output = named_outputs.get('output_port',
                                             'out_{}'.format(self.order)) 

    def generate_code(self): 
         return f"{self.output} = 0"
