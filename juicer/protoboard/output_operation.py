# -*- coding: utf-8 -*-

#import json
#import time
#from random import random
from textwrap import dedent

from juicer.operation import Operation

class LedOperation(Operation): 
    """
    Show value binary operation.  
    """
    COLOR_PARAM = "color"
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.color = parameters.get(self.COLOR_PARAM, 'red')
        self.has_code = len(self.named_inputs) > 0

    def generate_code(self): 
        #import pdb;pdb.set_trace()
        input_data1 = self.named_inputs['led_port']
        #code = "{in1}".format(in1=input_data1)
        #return dedent(code)
        #import pdb; pdb.set_trace()
        task_id = self.parameters.get('task').get('id')
        operation_id = self.parameters.get('task').get('operation').get('id')
        html = f'<div style="background: {self.color}; border-radius:10px; width:20px; height:20px">&nbsp;</div>'
        title = "Exemplo - LED"
        code = dedent(f"""
            if {input_data1}: 
                message='{html}'
            else: 
                message='off'    
            emit_event(
                        'update task', status='COMPLETED',
                        identifier='{task_id}',
                        message=message,
                        type='HTML', title='{title}',
                        task={{'id': '{task_id}'}},
                        operation={{'id': {operation_id}}},
                        operation_id={operation_id})
            """)
        return code
