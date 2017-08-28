# -*- coding: utf-8 -*-
import decimal
import itertools
import json
from textwrap import dedent

import datetime

from juicer.operation import Operation
from juicer.util import chunks
from juicer.util import dataframe_util
from juicer.util.dataframe_util import get_csv_schema

COLORS_PALETTE = list(reversed([
    '#590400', '#720601', '#8F0701', '#9C241E', '#AD443F',  # red
    '#934400', '#BD5700', '#ED6E00', '#FF9030', '#FFA75C',  # orange
    '#936300', '#BD7F00', '#ED9F00', '#FFBB30', '#FFCA5C',  # yellow
    '#285900', '#347201', '#428F01', '#579C1E', '#71AD3F',  # green
    '#005559', '#016D72', '#01898F', '#1E969C', '#3FA8AD',  # blue
    '#072163', '#0C2D7F', '#113A9F', '#3054AD', '#506FBB', ]))  # purple
SHAPES = ['diamond', 'point', 'circle']


def get_caipirinha_config(config):
    limonero_conf = config['juicer']['services']['limonero']
    caipirinha_conf = config['juicer']['services']['caipirinha']
    return dedent("""
    # Basic information to connect to other services
    config = {{
        'juicer': {{
            'services': {{
                'limonero': {{
                    'url': '{limonero_url}',
                    'auth_token': '{limonero_token}'
                }},
                'caipirinha': {{
                    'url': '{caipirinha_url}',
                    'auth_token': '{caipirinha_token}',
                    'storage_id': {storage_id}
                }},
            }}
        }}
    }}""".format(
        limonero_url=limonero_conf['url'],
        limonero_token=limonero_conf['auth_token'],
        caipirinha_url=caipirinha_conf['url'],
        caipirinha_token=caipirinha_conf['auth_token'],
        storage_id=caipirinha_conf['storage_id'], )
    )


class PublishVisualizationOperation(Operation):
    """
    This operation receives one dataframe as input and one or many
    VisualizationMethodOperation and persists the transformed data
    (currently HBase) for forthcoming visualizations
    """
    TITLE_PARAM = 'title'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.title = parameters.get(self.TITLE_PARAM, '')
        self.has_code = len(self.named_inputs) == 1
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

    def get_output_names(self, sep=", "):
        return ''

    @property
    def get_inputs_names(self):
        if isinstance(self.named_inputs['visualizations'], (list, tuple)):
            return ', '.join(self.named_inputs['visualizations'])
        else:
            return ', '.join([self.named_inputs['visualizations']])

    def generate_code(self):
        # Create connection with storage, get visualization table and initialize
        # list for visualizations metadata
        code_lines = [
            "from juicer.service import caipirinha_service",
            "from juicer.util.dataframe_util import CustomEncoder",
            "visualizations = []"
        ]
        if isinstance(self.named_inputs['visualizations'], (list, tuple)):
            visualizations = self.named_inputs['visualizations']
        else:
            visualizations = [self.named_inputs['visualizations']]

        for vis_model in visualizations:
            code_lines.append(dedent("""
            visualizations.append({{
                'job_id': '{job_id}',
                'task_id': {vis_model}.task_id,
                'title': {vis_model}.title ,
                'type': {{
                    'id': {vis_model}.type_id,
                    'name': {vis_model}.type_name
                }},
                'data': json.dumps({vis_model}.get_data(),
                               cls=CustomEncoder),
                'model': {vis_model}
            }})
            """).format(job_id=self.parameters['job_id'], vis_model=vis_model))

        # Register this new dashboard with Caipirinha
        code_lines.append(get_caipirinha_config(self.config))
        code_lines.append(dedent(u"""
            caipirinha_service.new_dashboard(config, '{title}', {user},
                {workflow_id}, '{workflow_name}',
                {job_id}, '{task_id}', visualizations, emit_event)
            """.format(
            title=self.title or 'Result for job ' + str(
                self.parameters.get('job_id', '0')),
            user=self.parameters['user'],
            workflow_id=self.parameters['workflow_id'],
            workflow_name=self.parameters['workflow_name'],
            job_id=self.parameters['job_id'],
            task_id=self.parameters['task']['id']
        )))

        code = '\n'.join(code_lines)
        return dedent(code)


####################################################
# Visualization operations used to generate models #
####################################################

class VisualizationMethodOperation(Operation):
    TITLE_PARAM = 'title'
    COLUMN_NAMES_PARAM = 'column_names'
    ORIENTATION_PARAM = 'orientation'
    ID_ATTR_PARAM = 'id_attribute'
    VALUE_ATTR_PARAM = 'value_attribute'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        # TODO: validate parameters
        self.title = parameters.get(
            self.TITLE_PARAM, 'Result for job ' + str(
                self.parameters.get('job_id', '0')))
        self.column_names = parameters.get(self.COLUMN_NAMES_PARAM, [])
        self.orientation = parameters.get(self.ORIENTATION_PARAM, '')
        self.id_attribute = parameters.get(self.ID_ATTR_PARAM, [])
        self.value_attribute = parameters.get(self.VALUE_ATTR_PARAM, [])

        # Visualizations are not cached!
        self.supports_cache = False
        self.output = self.named_outputs.get('visualization',
                                             'vis_task_{}'.format(self.order))

    def get_model_parameters(self):
        result = {}
        valid = ['x_axis_attribute', "y_title", "y_prefix", 'legend',
                 "y_suffix", "y_format", "x_title", "x_prefix", "x_suffix",
                 "x_format", "x_format", 'type',
                 'z_axis_attribute', 'z_title', 'z_prefix', 'z_suffix',
                 'z_format',
                 't_axis_attribute', 't_title', 't_prefix', 't_suffix',
                 't_format', ]
        for k, v in self.parameters.items():
            if k in valid:
                result[k] = v
        return result

    def get_output_names(self, sep=','):
        return self.output

    def get_model_name(self):
        NotImplementedError("Method generate_code should be implemented "
                            "in {} subclass".format(self.__class__))

    def generate_code(self):
        code_lines = [dedent(
            u"""
            from juicer.spark.vis_operation import {model}
            from juicer.util.dataframe_util import CustomEncoder

            {out} = {model}(
                {input}, '{task}', '{op}',
                '{op_slug}', '{title}',
                {columns},
                '{orientation}', {id_attr}, {value_attr},
                params={params})
            """.format(out=self.output,
                       model=self.get_model_name(),
                       input=self.named_inputs['input data'],
                       task=self.parameters['task']['id'],
                       op=self.parameters['operation_id'],
                       op_slug=self.parameters['operation_slug'],
                       title=self.title,
                       columns=json.dumps(self.column_names),
                       orientation=self.orientation,
                       id_attr=self.id_attribute,
                       value_attr=self.value_attribute,
                       params=json.dumps(self.get_model_parameters() or {}),
                       ))]
        if len(self.named_outputs) == 0:
            # Standalone visualization, without a dashboard
            code_lines.append("from juicer.service import caipirinha_service")
            code_lines.append(get_caipirinha_config(self.config))
            code_lines.append(dedent("""
            visualization = {{
                'job_id': '{job_id}',
                'task_id': {out}.task_id,
                'title': {out}.title ,
                'type': {{
                    'id': {out}.type_id,
                    'name': {out}.type_name
                }},
                'model': {out},
                'data': json.dumps({out}.get_data(),
                               cls=CustomEncoder),
            }}""").format(job_id=self.parameters['job_id'],
                          out=self.output))

            code_lines.append(dedent(u"""
            caipirinha_service.new_visualization(
                config,
                {user},
                {workflow_id}, {job_id}, '{task_id}',
                visualization, emit_event)
            """.format(
                user=self.parameters['user'],
                workflow_id=self.parameters['workflow_id'],
                job_id=self.parameters['job_id'],
                task_id=self.parameters['task']['id']
            )))
        return '\n'.join(code_lines)


class BarChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'BarChartModel'


class PieChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'PieChartModel'


class DonutChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'DonutChartModel'


class LineChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'LineChartModel'


class AreaChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'AreaChartModel'


class TableVisualizationOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'TableVisualizationModel'


class ScatterPlotOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'ScatterPlotModel'


class MapOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'MapModel'


class SummaryStatisticsOperation(VisualizationMethodOperation):
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        if not parameters.get(self.TITLE_PARAM):
            parameters[self.TITLE_PARAM] = 'Summary'
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, None)

    def get_model_parameters(self):
        return {self.ATTRIBUTES_PARAM: self.attributes or []}

    def get_model_name(self):
        return 'SummaryStatisticsModel'


#######################################################
# Visualization Models used inside the code generated #
#######################################################

class VisualizationModel(object):
    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation,
                 id_attribute, value_attribute, params):
        self.data = data
        self.task_id = task_id
        self.type_id = type_id
        self.type_name = type_name
        self.title = title
        self.column_names = column_names
        self.orientation = orientation
        self.params = params

        if len(id_attribute) > 0 and isinstance(id_attribute, list):
            self.id_attribute = id_attribute[0]
        else:
            self.id_attribute = id_attribute

        self.value_attribute = value_attribute

    def get_data(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def get_schema(self):
        return self.data.schema.json()

    def get_icon(self):
        return 'fa-question-o'

    def get_column_names(self):
        return ""


class ChartVisualization(VisualizationModel):
    def get_data(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def _get_title_legend_tootip(self):
        """ Common title and legend """
        return {
            "title": self.title,
            "legend": {
                "isVisible": True,
                "text": "{{name}}"
            },
            "tooltip": {
                "title": [
                    "{{name}}"
                ],
                "body": [
                    "<span class='metric'>FIXME</span>"
                    "<span class='number'>{{name}}</span>"
                ]
            },
        }

    def _get_axis_info(self):
        schema = self.data.schema
        if not self.params.get('x_axis_attribute'):
            raise ValueError('X-axis attribute not specified')
        x = self.params.get('x_axis_attribute')[0]
        x_attr = [c for c in schema if c.name == x]
        y_attrs = [c for c in schema if c.name in self.column_names]
        if len(x_attr):
            x_attr = x_attr[0]
        else:
            raise ValueError(
                'Attribute {} for X-axis does not exist in ({})'.format(
                    x, ', '.join([c.name for c in schema])))
        if len(y_attrs) == 0:
            raise ValueError(
                'At least one attribute for Y-axis does not exist: {}'.format(
                    ', '.join(self.params.get('column_names'))))

        # @FIXME: Improve this code with other data types
        if x_attr.dataType.jsonValue() == 'date':
            x_type = 'date'
        elif x_attr.dataType.jsonValue() == 'boolean':
            x_type = 'bool'
        elif x_attr.dataType.jsonValue() == 'timestamp':
            x_type = 'timestamp'
        elif x_attr.dataType.jsonValue() == 'string':
            x_type = 'string'
        else:
            x_type = 'number'
        return x_attr, x_type, y_attrs

    @staticmethod
    def _format(value):
        if any([isinstance(value, datetime.datetime),
                isinstance(value, datetime.date)]):
            return value.isoformat()
        elif isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return value


class BarChartModel(ChartVisualization):
    """ Bar chart model for visualization of data """

    def get_icon(self):
        return 'fa-bar-chart'

    def get_data(self):
        x_attr, x_type, y_attrs = self._get_axis_info()

        rows = self.data.collect()

        colors = {}
        color_counter = 0
        for i, attr in enumerate(y_attrs):
            color = COLORS_PALETTE[(i % 6) * 5 + ((i / 6) % 5)]
            colors[attr.name] = {
                'fill': color,
                'gradient': color,
                'stroke': color,
            }
            color_counter = i

        result = {}
        result.update(self._get_title_legend_tootip())

        result.update({
            "x": {
                "title": self.params.get("x_title"),
                "type": x_type,
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
            },
            "y": {
                "equal": True,
                "title": self.params.get("y_title"),
                "prefix": self.params.get("y_prefix"),
                "suffix": self.params.get("y_suffix"),
                "format": self.params.get("y_format", {}).get('key'),
            },
            "data": []

        })
        if x_type in ['number']:
            result['x']['format'] = self.params.get("x_format", {}).get('key')
        elif x_type in ['timestamp', 'date']:
            result['x']["outFormat"] = self.params.get("x_format", {}).get(
                'key')
            result['x']["inFormat"] = self.params.get("x_format", {}).get('key')

        for inx_row, row in enumerate(rows):
            x_value = row[x_attr.name]
            if x_value not in colors:
                inx_row += 1
                color = COLORS_PALETTE[(color_counter % 6) * 5 +
                                       ((color_counter / 6) % 5)]
                colors[x_value] = {
                    'fill': color,
                    'gradient': color,
                    'stroke': color,
                }

            data = {
                'x': LineChartModel._format(x_value),
                'name': row[x_attr.name],
                'key': row[x_attr.name],
                'color': COLORS_PALETTE[
                    (inx_row % 6) * 5 + ((inx_row / 6) % 5)],
                'values': []
            }
            result['data'].append(data)
            for i, attr in enumerate(y_attrs):
                data['values'].append(
                    {
                        'x': attr.name,
                        'name': attr.name,
                        'y': LineChartModel._format(row[attr.name]),
                    }
                )
                if i >= 100:
                    raise ValueError(
                        'The maximum number of values for x-axis is 100.')

        result['colors'] = colors
        return result


class PieChartModel(ChartVisualization):
    """
    In PieChartModel, x_attr contains the label and y_attrs[0] contém os valores
    """

    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation, id_attribute, value_attribute, params):
        ChartVisualization.__init__(self, data, task_id, type_id,
                                    params.get('type', 'pie-chart'),
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return 'fa-pie-chart'

    def _get_axis_info(self):
        schema = self.data.schema

        if self.id_attribute:
            label = self.id_attribute
        else:
            label = self.value_attribute

        value_attr = [c for c in schema if c.name in self.value_attribute[0]]
        if len(value_attr):
            value_attr = value_attr[0]
        else:
            raise ValueError(
                'Attribute {} does not exist in ({})'.format(label, ', '.join(
                    [c.name for c in schema])))

        label_attr = [c for c in schema if c.name == label]
        if len(label_attr):
            label_attr = label_attr[0]
        else:
            raise ValueError(
                'Attribute {} for label does not exist in ({})'.format(
                    label, ', '.join([c.name for c in schema])))

        return label_attr, None, value_attr

    def get_data(self):
        label_attr, _, value_attr = self._get_axis_info()

        rows = self.data.collect()
        result = self._get_title_legend_tootip()
        result['legend']['isVisible'] = self.params.get('legend') in ('1', 1)

        result.update({
            "x": {
                "title": self.params.get("x_title"),
                "value": "sum",
                "color": "#222",
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
                "format": self.params.get("x_format", {}).get('key'),
            },
            "data": []

        })
        for i, row in enumerate(rows):
            data = {
                'x': float(row[value_attr.name]),
                'value': float(row[value_attr.name]),
                'id': '{}_{}'.format(label_attr.name, i),
                'name': row[label_attr.name],
                'label': row[label_attr.name],
                'color': COLORS_PALETTE[(i % 6) * 5 + ((i / 6) % 5)],
            }
            result['data'].append(data)
            if i >= 100:
                raise ValueError(
                    'The maximum number of values for this chart is 100.')
        return result


class DonutChartModel(PieChartModel):
    def get_data(self):
        data = super(DonutChartModel, self).get_data()
        return data


class LineChartModel(ChartVisualization):
    def get_icon(self):
        return 'fa-line-chart'

    def get_data(self):
        x_attr, x_type, y_attrs = self._get_axis_info()

        rows = self.data.collect()

        data = []
        for i, attr in enumerate(y_attrs):
            data.append({
                "id": attr.name,
                "name": attr.name,
                "color": COLORS_PALETTE[(i % 6) * 5 + ((i / 6) % 5)],
                "pointColor": COLORS_PALETTE[(i % 6) * 5 + ((i / 6) % 5)],
                "pointShape": SHAPES[i % len(SHAPES)],
                "pointSize": 8,
                "values": []
            })

        result = {}
        result.update(self._get_title_legend_tootip())

        result.update({
            "y": {
                "title": self.params.get("y_title"),
                "prefix": self.params.get("y_prefix"),
                "suffix": self.params.get("y_suffix"),
                "format": self.params.get("y_format", {}).get('key'),
            },
            "x": {
                "title": self.params.get("x_title"),
                "type": x_type,
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
            },
            "data": data
        })
        if x_type in ['number']:
            result['x']['format'] = self.params.get("x_format", {}).get('key')
        elif x_type in ['timestamp', 'date']:
            result['x']["outFormat"] = self.params.get("x_format", {}).get(
                'key')
            result['x']["inFormat"] = self.params.get("x_format", {}).get('key')

        for row in rows:
            for i, attr in enumerate(y_attrs):
                data[i]['values'].append(
                    {
                        "x": LineChartModel._format(row[x_attr.name]),
                        "y": LineChartModel._format(row[attr.name]),
                    }
                )
        return result


class AreaChartModel(LineChartModel):
    def get_icon(self):
        return 'fa-area-chart'


class ScatterPlotModel(ChartVisualization):
    """
    Scatter plot chart model
    """

    @staticmethod
    def _get_attr_type(attr):
        # @FIXME: Improve this code with other data types
        if attr.dataType.jsonValue() == 'date':
            attr_type = 'date'
        elif attr.dataType.jsonValue() == 'boolean':
            attr_type = 'bool'
        elif attr.dataType.jsonValue() == 'timestamp':
            attr_type = 'timestamp'
        elif attr.dataType.jsonValue() == 'string':
            attr_type = 'string'
        else:
            attr_type = 'number'

        return attr_type

    def get_data(self):
        schema = self.data.schema

        result = {}
        attrs = {}
        for axis in ['x', 'y', 'z', 't']:
            name = self.params.get('x_axis_attribute', [None])[0]
            attrs[axis] = [c for c in schema if c.name == name]
            if attrs[axis]:
                result[axis] = {
                    "title": self.params.get("{}_title".format(axis)),
                    "prefix": self.params.get("{}_prefix".format(axis)),
                    "suffix": self.params.get("{}_suffix".format(axis)),
                    "format": self.params.get("{}_format".format(axis)),
                    "outFormat": self.params.get("{}_format".format(axis)),
                    "inFormat": self.params.get("{}_format".format(axis)),
                    "type": self._get_attr_type(attrs[axis])
                }
        rows = self.data.collect()

        data = []
        for i, attr in enumerate(y_attrs):
            data.append({
                "id": attr.name,
                "name": attr.name,
                "color": COLORS_PALETTE[(i % 6) * 5 + ((i / 6) % 5)],
                "pointColor": COLORS_PALETTE[(i % 6) * 5 + ((i / 6) % 5)],
                "pointShape": SHAPES[i % len(SHAPES)],
                "pointSize": 8,
                "values": []
            })

        result = {}
        result.update(self._get_title_legend_tootip())

        result.update({
            "y": {
                "title": self.params.get("y_title"),
                "prefix": self.params.get("y_prefix"),
                "suffix": self.params.get("y_suffix"),
                "format": self.params.get("y_format", {}).get('key'),
            },
            "x": {
                "title": self.params.get("x_title"),
                "type": x_type,
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
                "outFormat": self.params.get("x_format", {}).get('key'),
                "inFormat": self.params.get("x_format", {}).get('key'),
            },
            "data": data

        })
        for row in rows:
            for i, attr in enumerate(y_attrs):
                data[i]['values'].append(
                    {
                        "x": LineChartModel._format(row[x_attr.name]),
                        "y": LineChartModel._format(row[attr.name]),
                    }
                )
        return result


class HtmlVisualizationModel(VisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data, task_id, type_id, type_name, title,
                 column_names,
                 orientation, id_attribute, value_attribute, params):
        # type_id = 1
        # type_name = 'html'
        VisualizationModel.__init__(self, data, task_id, type_id, type_name,
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return "fa-html5"

    def get_data(self):
        return self.data

    def get_schema(self):
        return ''


class TableVisualizationModel(VisualizationModel):
    def __init__(self, data, task_id, type_id, type_name, title,
                 column_names,
                 orientation, id_attribute, value_attribute, params):
        type_id = 35
        type_name = 'table-visualization'
        if not title:
            title = 'Results'
        VisualizationModel.__init__(self, data, task_id, type_id, type_name,
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return 'fa-table'

    def get_data(self):
        """
        Returns data as tabular (list of lists in Python).
        """
        if self.column_names:
            rows = self.data.limit(50).select(*self.column_names).rdd.map(
                dataframe_util.convert_to_python).collect()
        else:
            rows = self.data.limit(50).rdd.map(
                dataframe_util.convert_to_python).collect()

        return {"rows": rows,
                "attributes": self.get_column_names().split(','),
                "schema": self.get_schema()}

    def get_column_names(self):
        if self.column_names:
            return ','.join(self.column_names)
        else:
            return get_csv_schema(self.data, only_name=True)


class SummaryStatisticsModel(TableVisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data, task_id, type_id, type_name, title,
                 column_names,
                 orientation, id_attribute, value_attribute, params):
        TableVisualizationModel.__init__(self, data, task_id, type_id,
                                         type_name,
                                         title, column_names, orientation,
                                         id_attribute, value_attribute,
                                         params)
        self.names = ''
        self.numeric_attrs = [
            t[0] for t in self.data.dtypes
            if t[1] in ['int', 'double', 'tinyint',
                        'bigint', 'smallint'] or t[1][:7] == 'decimal']

        all_attr = [t[0] for t in self.data.dtypes]
        if len(self.params['attributes']) == 0:
            self.attrs = all_attr
        else:
            self.attrs = [attr for attr in all_attr if
                          attr in self.params['attributes']]
        self.names = ['attribute', 'max', 'min', 'stddev', 'count', 'avg',
                      'approx. distinct', 'missing']

        self.names.extend(
            ['correlation to {}'.format(attr) for attr in self.attrs])

        self.column_names = self.names

    def get_icon(self):
        return 'fa-table'

    # noinspection PyUnresolvedReferences
    def get_data(self):
        """
        Returns statistics about attributes in a data frame
        """

        from pyspark.sql import functions

        # Correlation pairs
        corr_pairs = list(
            chunks(
                list(itertools.product(self.attrs, self.attrs)),
                len(self.attrs)))

        # Cache data
        self.data.cache()

        df_count = self.data.count()

        # TODO: Implement median using df.approxQuantile('col', [.5], .25)

        stats = []
        for i, name in enumerate(self.attrs):
            df_col = functions.col(name)
            stats.append(functions.lit(name))
            stats.append(functions.max(df_col).alias('max_{}'.format(name)))
            stats.append(functions.min(df_col).alias('min_{}'.format(name)))
            if name in self.numeric_attrs:
                stats.append(functions.round(
                    functions.stddev(df_col), 4).alias(
                    'stddev_{}'.format(name)))
            else:
                stats.append(functions.lit('-'))
            stats.append(
                functions.count(df_col).alias('count_{}'.format(name)))
            if name in self.numeric_attrs:
                stats.append(functions.round(
                    functions.avg(df_col), 4).alias('avg_{}'.format(name)))
            else:
                stats.append(functions.lit('-'))

            stats.append(functions.approx_count_distinct(df_col).alias(
                'distinct_{}'.format(name)))
            stats.append((df_count - functions.count(df_col)).alias(
                'missing_{}'.format(name)))
            for pair in corr_pairs[i]:
                if all([pair[0] in self.numeric_attrs,
                        pair[1] in self.numeric_attrs]):
                    stats.append(
                        functions.round(functions.corr(*pair), 4).alias(
                            'corr_{}'.format(i)))
                else:
                    stats.append(functions.lit('-'))

        aggregated = self.data.agg(*stats).take(1)[0]
        n = len(self.names)
        rows = [aggregated[i:i + n] for i in range(0, len(aggregated), n)]

        return {"rows": rows, "attributes": self.get_column_names().split(','),
                "schema": self.get_schema()}
