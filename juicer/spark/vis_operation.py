# -*- coding: utf-8 -*-
import json
from textwrap import dedent

import datetime

from juicer.operation import Operation, ResultType
from juicer.runner import configuration
from juicer.service import limonero_service
from juicer.util import dataframe_util
import time
import urlparse
import uuid


class PublishVisOperation(Operation):
    """
    This operation receives one dataframe as input and one or many
    VisualizationMethodOperation and persists the transformed data
    (currently HBase) for forthcoming visualizations
    """
    TITLE_PARAM = 'title'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.config = configuration.get_config()
        self.title = parameters.get(self.TITLE_PARAM, '')
        self.has_code = len(self.inputs) > 1
        self.supports_cache = False
        self.icon = 'fa-question'

    """
    This operation represents a strategy for visualization and is used together
    with 'PublishVisOperation' to create a visualization dashboard
    """

    def get_generated_results(self):
        return []
        # return [
        #     {'type': ResultType.VISUALIZATION,
        #      'id': self.parameters['task']['id'],
        #      'icon': self.icon,
        #      'title': self.title,
        #      }
        # ]

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['data'],
                          self.named_inputs['visualization']])

    def generate_code(self):

        # Get Limonero configuration
        limonero_config = self.config['juicer']['services']['limonero']

        # Get Caipirinha configuration
        caipirinha_config = self.config['juicer']['services']['caipirinha']

        # Storage refers to the underlying environment used for storing
        # visualizations, e.g., Hbase
        storage = limonero_service.get_storage_info(
            limonero_config['url'],
            str(limonero_config['auth_token']), caipirinha_config['storage_id'])

        # Get hbase hostname and port
        parsed_url = urlparse.urlparse(storage['url'])
        host = parsed_url.hostname
        port = parsed_url.port

        # Create connection with storage, get visualization table and initialize
        # list for visualizations metadata
        code_lines = [
            "import happybase",
            "from juicer.service import caipirinha_service",
            "connection = happybase.Connection(host='{}', port={})".format(
                host, port),
            "vis_table = connection.table('visualization')",
            "visualizations = []"]

        # The following code template will, for each visualization method:
        #
        # [1] Transform data (and schema) according to the visualization method
        # [2] Take task_id (uuid) and job_id (int) to identify the dashboard
        # [3] Register a timestamp for the visualization
        # [4] Insert this particular visualization into hbase
        # [5] Append the visualization model for later persistency of metadata
        #
        # [Considerations concerning Hbase storage schema]
        #   - We chose to encode data from all columns as json.
        #   - That means that we assume that every data can be encoded as json,
        #   including 'cf:data' which is supposed to contain the actual
        #   visualization data.
        vis_code_tmpl = \
            """
            # HBase value composed by several columns, the last one refers
            # to the visualization data
            vis_value = {{
                b'cf:user': json.dumps({{'name': '{user}', 'id': {user_id} }}),
                b'cf:workflow': json.dumps({{'id': {workflow_id} }}),
                b'cf:title': {vis_model}.title,
                b'cf:column_names': {vis_model}.column_names,
                b'cf:orientation': {vis_model}.orientation,
                b'cf:attributes': json.dumps({{ 'id': {vis_model}.id_attribute,
                        'value': {vis_model}.value_attribute }}),
                b'cf:data': json.dumps({vis_model}.get_data({dfdata})),
                b'cf:schema': {dfdata}.schema.json()
            }}
            vis_table.put(b'{job_id}-' + {task_id}, vis_value,
                timestamp=int(time.time()))
            visualizations.append({{
                'job_id': '{job_id}',
                'task_id': {task_id},
                'title': {title} ,
                'type': {{
                    'id': {vis_model}.type_id,
                    'name': {vis_model}.type_name
                }}
            }})
            emit_event('task result', status='COMPLETED',
                identifier={task_id}, msg='Result generated',
                type='VISUALIZATION', title={vis_model}.title,
                task={{'id': {task_id} }},
                operation={{'id': {vis_model}.type_id }},
                operation_id={vis_model}.type_id)
            """
        for vis_model in self.inputs:
            # NOTE: For now we assume every other input but 'data' are
            # visualizations strategies that will compose the dashboard
            if vis_model != self.named_inputs['data']:
                code_lines.append(dedent(vis_code_tmpl.format(
                    job_id=self.parameters['job_id'],
                    task_id='{}.task_id'.format(vis_model),
                    user_id=self.parameters['user']['id'],
                    user=self.parameters['user']['login'],
                    workflow_id=self.parameters['workflow_id'],
                    dfdata=self.named_inputs['data'],
                    vis_model=vis_model,
                    vis_type_id=self.parameters['operation_id'],
                    vis_type_name=self.parameters['operation_slug'],
                    title='{}.title'.format(vis_model)
                )))

        # Register this new dashboard with Caipirinha
        code_lines.append("""caipirinha_service.new_dashboard('{base_url}',
            '{token}', '{title}', {user},
            {workflow_id}, '{workflow_name}',
            {job_id}, '{task_id}', visualizations)""".format(
            base_url=caipirinha_config['url'],
            token=caipirinha_config['auth_token'],
            title=self.title or 'Result for job' + self.parameters.get('job_id',
                                                                       '0'),
            user_name=self.parameters['user']['name'],
            user=self.parameters['user'],
            workflow_id=self.parameters['workflow_id'],
            workflow_name=self.parameters['workflow_name'],
            job_id=self.parameters['job_id'],
            task_id=self.parameters['task']['id']
        ))

        # Make sure we close the hbase connection
        code_lines.append('connection.close()')

        # No return
        code_lines.append('{} = None'.format(self.output))

        code = '\n'.join(code_lines)
        return dedent(code)


####################################################
# Visualization operations used to generate models #
####################################################

class VisualizationMethodOperation(Operation):
    def generate_code(self):
        NotImplementedError("Method generate_code should be implemented "
                            "in {} subclass".format(self.__class__))

    TITLE_PARAM = 'title'
    COLUMN_NAMES_PARAM = 'column_names'
    ORIENTATION_PARAM = 'orientation'
    ID_ATTR_PARAM = 'id_attribute'
    VALUE_ATTR_PARAM = 'value_attribute'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        # TODO: validate parameters
        self.title = parameters.get(self.TITLE_PARAM, '')
        self.column_names = parameters.get(self.COLUMN_NAMES_PARAM, '')
        self.orientation = parameters.get(self.ORIENTATION_PARAM, '')
        self.id_attribute = parameters.get(self.ID_ATTR_PARAM, [])
        self.value_attribute = parameters.get(self.VALUE_ATTR_PARAM, [])

        # Visualizations are not cached!
        self.supports_cache = False

    def get_output_names(self, sep=','):
        return self.output


class BarChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, inputs,
                                              outputs, named_inputs,
                                              named_outputs)

    def generate_code(self):
        code = \
            """
            from juicer.spark.vis_operation import BarChartModel
            {} = BarChartModel('{}','{}', '{}','{}','{}','{}',{},{})
            """.format(self.output,
                       self.parameters['task']['id'],
                       self.parameters['operation_id'],
                       self.parameters['operation_slug'],
                       self.title, self.column_names, self.orientation,
                       self.id_attribute, self.value_attribute)
        return dedent(code)


class PieChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, inputs,
                                              outputs, named_inputs,
                                              named_outputs)

    def generate_code(self):
        code = \
            """
            from juicer.spark.vis_operation import PieChartModel
            {} = PieChartModel('{}','{}', '{}', '{}','{}','{}','{}',{})
            """.format(self.output,
                       self.parameters['task']['id'],
                       self.parameters['operation_id'],
                       self.parameters['operation_slug'],
                       self.title, self.column_names, self.orientation,
                       self.id_attribute, self.value_attribute)
        return dedent(code)


class LineChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, inputs,
                                              outputs, named_inputs,
                                              named_outputs)

    def generate_code(self):
        code = \
            """
            from juicer.spark.vis_operation import LineChartModel
            {} = LineChartModel('{}','{}', '{}', '{}','{}','{}','{}',{})
            """.format(self.output,
                       self.parameters['task']['id'],
                       self.parameters['operation_id'],
                       self.parameters['operation_slug'],
                       self.title, self.column_names, self.orientation,
                       self.id_attribute, self.value_attribute)
        return dedent(code)


class AreaChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, inputs,
                                              outputs, named_inputs,
                                              named_outputs)

    def generate_code(self):
        code = \
            """
            from juicer.spark.vis_operation import AreaChartModel
            {} = AreaChartModel('{}','{}', '{}', '{}','{}','{}','{}',{})
            """.format(self.output,
                       self.parameters['task']['id'],
                       self.parameters['operation_id'],
                       self.parameters['operation_slug'],
                       self.title, self.column_names, self.orientation,
                       self.id_attribute, self.value_attribute)
        return dedent(code)


class TableVisOperation(VisualizationMethodOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, inputs,
                                              outputs, named_inputs,
                                              named_outputs)

    def generate_code(self):
        code = \
            """
            from juicer.spark.vis_operation import TableVisModel
            {} = TableVisModel('{}','{}', '{}', '{}','{}','{}','{}', {})
            """.format(self.output,
                       self.parameters['task']['id'],
                       self.parameters['operation_id'],
                       self.parameters['operation_slug'],
                       self.title, self.column_names, self.orientation,
                       self.id_attribute, self.value_attribute)
        return dedent(code)


#######################################################
# Visualization Models used inside the code generated #
#######################################################

class VisualizationModel:
    def __init__(self, task_id, type_id, type_name, title, column_names,
                 orientation,
                 id_attribute, value_attribute):
        self.task_id = task_id
        self.type_id = type_id
        self.type_name = type_name
        self.title = title
        self.column_names = column_names
        self.orientation = orientation
        self.id_attribute = id_attribute
        self.value_attribute = value_attribute

    def get_data(self, data):
        # return data.rdd.map(dataframe_util.convert_to_csv).collect()
        # return data.rdd.map(dataframe_util.convert_to_python).collect()
        raise NotImplementedError('Should be implemented in derived classes')

    def get_schema(self, data):
        return dataframe_util.get_csv_schema(data)

    def get_dict_schema(self, data):
        return dataframe_util.get_dict_schema(data)

    def get_icon(self):
        return 'fa-question-o'


class BarChartModel(VisualizationModel):
    def get_data(self, data):
        return data.rdd.map(dataframe_util.convert_to_python).collect()

    def get_icon(self):
        return 'fa-bar-chart'


class PieChartModel(VisualizationModel):
    def get_icon(self):
        return 'fa-pie-chart'

    def get_data(self, data):
        """
        Returns data as a list dictionaries in Python (JSON encoder friendly).
        """
        return data.rdd.map(
            dataframe_util.format_row_for_visualization).collect()


class AreaChartModel(VisualizationModel):
    def get_icon(self):
        return 'fa-area-chart'

    def get_data(self, data):
        return data.rdd.map(dataframe_util.convert_to_python).collect()


class LineChartModel(VisualizationModel):
    def get_icon(self):
        return 'fa-line-chart'

    def get_data(self, data):
        return data.rdd.map(dataframe_util.convert_to_python).collect()


class TableVisModel(VisualizationModel):
    def get_icon(self):
        return 'fa-table'

    def get_data(self, data):
        """
        Returns data as tabular (list of lists in Python).
        """
        return data.rdd.map(dataframe_util.convert_to_python).collect()
