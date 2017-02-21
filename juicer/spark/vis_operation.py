# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation
from juicer.runner import configuration
from juicer.service import limonero_service
import time
import uuid

class PublishVisOperation(Operation):
    """
    This operation receives one dataframe as input and one or many
    VisualizationMethodOperation and persists the transformed data
    (currently HBase) for forthcoming visualizations
    """
    
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.config = configuration.get_config()
        self.has_code = len(self.inputs) > 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        
        # Get hbase storage info from Limonero through storage_id
        limonero_config = self.config['services']['limonero']
        storage = limonero_service.get_storage_info(
                limonero_config['url'],
                limonero_config['auth_token'], limonero_config['storage_id'])
        
        # Get hbase hostname and port
        parsed_url = urlparse.urlparse(storage['url'])
        host = parsed_url.hostname
        port = parsed_url.port

        # Create connection with storage, get visualization table and initialize
        # list for visualizations metadata
	code_lines = []
	code_lines.append("import happybase")
        code_lines.append("from juicer.service import caipirinha_service")
	code_lines.append("connection = happybase.Connection(host={}, port={})".format(host,port))
	code_lines.append("vis_table = connection.table('visualization')")
        code_lines.append("visualizations = []")

        # The following code template will, for each visualization method:
        # [1] Get dataframe data (collect)
        # [2] Transform the local data according to the visualization method
        # [3] Create a visualization id (uuid)
        # [4] Register a timestamp for the visualization
        # [5] Insert this particular visualization into hbase
        # [6] Append the visualization model for later persistency of metadata
        vis_code_tmpl = """vis_value = {{
            b'cf:user_id': '{user_id}',
            b'cf:user': '{user}',
            b'cf:workflow_id': '{workflow_id}',
            b'cf:{vis_model}': {vis_model}.transform_data({dfdata}.collect())
        }}
        vis_table.put(b'{vis_uuid}', vis_value, timestamp={vis_timestamp})
        visualizations.append({{
            'vis_uuid': {vis_uuid},
            'vis_type': {vis_model}.__class__.__name__,
            'vis_timestamp': {vis_timestamp},
            'title': {vis_model}.title,
            'labels': {vis_model}.labels,
            'orientation': {vis_model}.orientation,
            'id_attribute': {vis_model}.id_attribute,
            'value_attribute': {vis_model}.value_attribute
        }})"""
        for vismodel in self.inputs[1:]:
            vis_uuid = uuid.uuid4()
            vis_timestamp = int(time.time())
            code_lines.append(vis_code_tmpl.format(
                user_id=self.parameters['user']['id'],
                user=self.parameters['user']['login'],
                workflow_id=self.parameters['workflow_id'],
                dfdata=self.inputs[0],
                vis_model=vis_model,
                vis_uuid=vis_uuid,
                vis_timestamp=vis_timestamp
            ))
        
        # Get Caipirinha configuration
        caipirinha_config = self.config['services']['caipirinha']

        # Register this new dashboard with Caipirinha
        code_lines.append("""caipirinha_service.new_dashboard({base_url}, {token},
            "{user_name}'s dashboard of workflow '{workflow_name}'",
            {user_id}, {user_login}, {user_name},
            {workflow_id}, {workflow_name}, visualizations)""".format(
                base_url=caipirinha_config['url'],
                token=caipirinha_config['auth_token'],
                user_id=self.parameters['user']['id'],
                user_login=self.parameters['user']['login'],
                user_name=self.parameters['user']['name'],
                workflow_id=self.parameters['workflow_id'],
                workflow_name=self.parameters['workflow_name']
            ))

        # Make sure we close the hbase connection
	code_lines.append("connection.close()")

        code = '\n'.join(vis_codes)
        return dedent(code)

class VisualizationMethodOperation(Operation):
    """
    This operation represents a strategy for visualization and is used together
    with 'PublishVisOperation' to create a visualization dashboard
    """
    TITLE_PARAM = 'title'
    LABELS_PARAM = 'labels'
    ORIENTATION_PARAM = 'orientation'
    ID_ATTR_PARAM = 'id_attribute'
    VALUE_ATTR_PARAM = 'value_attribute'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        # TODO: validate parameters
        self.title = parameters.get(self.TITLE_PARAM, '')
        self.labels = parameters.get(self.LABELS_PARAM, '')
        self.orientation = parameters.get(self.ORIENTATION_PARAM, '')
        self.id_attribute = parameters.get(self.ID_ATTR_PARAM, [])
        self.value_attribute = parameters.get(self.VALUE_ATTR_PARAM, [])
    
    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def transform_data(self, data):
        raise NotImplementedError(
            "Transformation method is not implemented (%s)" % \
            type(self).__name__)

class BarChartOperation(VisualizationMethodOperation):
    
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

    def generate_code(self):
	code = """from juicer.spark.vis_operation import BarChartOperation
            {} = BarChartOperation({},{},{},{},{})""".format(self.output,
                    self.parameters, self.inputs, self.outputs,
                    self.named_inputs, self.named_outputs)
        return dedent(code)

    def transform_data(self, data):
        # TODO: transform data accordingly
        return data
