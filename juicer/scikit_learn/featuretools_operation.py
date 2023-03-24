# -*- coding: utf-8 -*-

from juicer.operation import Operation
from textwrap import dedent

class DerivateNewColumnsOperation(Operation): 
    """
    Derivate new columns using feature tools; 
    Parameters:  
    - TRANSACTION_INDEX:Transaction index name;
    - TRANSACTION_TIME: Transaction time name; 
    - RELATIONSHIP_INDEX: Relationship dataframe index; 
    - AGGREGATION_PRIM: Aggreation primitive; 
    - TRANSFORM_PRIM: Transform primitive;
    """
    TRANSACTION_INDEX  = 'transaction-index'
    TRANSACTION_TIME   = 'transaction-time'  
    RELATIONSHIP_INDEX = 'relationship-dataframe-index'
    AGGREGATION_PRIM   = 'aggregation-primitives'
    TRANSFORM_PRIM     = 'transform-primitives'

    #Declarate parameters 
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        #Import libraries
        self.transpiler_utils.add_import('import pandas as pd') 
        self.transpiler_utils.add_import('import featuretools as ft') 

        #Attibute-selector is a list with one element 
        self.transaction_index      = parameters.get(self.TRANSACTION_INDEX)[0] 
        self.transaction_time       = parameters.get(self.TRANSACTION_TIME)[0] 
        self.relationship_index     = parameters.get(self.RELATIONSHIP_INDEX)[0] 
        self.aggregation_primitive  = parameters.get(self.AGGREGATION_PRIM) 
        self.transform_primitive    = parameters.get(self.TRANSFORM_PRIM) 

        self.has_code = (len(self.named_inputs)==1 and len(self.named_outputs) > 0)
        self.output = self.named_outputs.get('output_port',
                                             'out_{}'.format(self.order)) 
        self.input = self.named_inputs.get('input_port')

    def generate_code(self): 
        code = dedent(f"""
        es = ft.EntitySet(id="entityset_data")      

        es = es.add_dataframe(
                dataframe_name='transaction',
                dataframe={self.input},
                index='{self.transaction_index}',
                time_index='{self.transaction_time}')
        
        es = es.normalize_dataframe(
                base_dataframe_name='transaction',
                new_dataframe_name='relationship',
                index='{self.relationship_index}')

        feature_matrix, features = ft.dfs(entityset=es,
                target_dataframe_name='relationship', 
                agg_primitives={self.aggregation_primitive},
                trans_primitives={self.transform_primitive},
                verbose=false)

        {self.output} = feature_matrix        
        """)

        return code 
