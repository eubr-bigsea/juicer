# -*- coding: utf-8 -*-

#operations  
import juicer.protoboard.input_operation as inp
import juicer.protoboard.ci_operation as ci 
import juicer.protoboard.output_operation as out

#Standard
import os 
from juicer import operation 
from juicer.transpiler import Transpiler 

class ProtoboardTranspiler(Transpiler): 
   """
     Convert Lemonade workflow representation (JSON) into code to be run in 
     Protoboard
   """
   def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
       super(ProtoboardTranspiler, self).__init__(
           configuration, os.path.abspath(os.path.dirname(__file__)),
           slug_to_op_id, port_id_to_port)

       self._assign_operations()
 
   def _assign_operations(self):
     input_ops = { 
            'fonte-5v': inp.Fonte5vOperation, 
            'terra': inp.TerraOperation,  
     } 

     ci_ops = { 
            'and-7408': ci.AndOperation,
     } 

     output_ops = {  
             'led': out.LedOperation, 
     } 

     self.operations = {}  
     for ops in [input_ops, ci_ops, output_ops]:
        self.operations.update(ops)
