# coding=utf-8


import collections
import decimal
import itertools
import json
from collections import Iterable
from textwrap import dedent

import datetime

from juicer import auditing
from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util import chunks
from juicer.util import dataframe_util
from juicer.util.dataframe_util import get_csv_schema

TRUE_VALS = [True, 1, '1']

COLORS_PALETTE = list(reversed([
    '#590400', '#720601', '#8F0701', '#9C241E', '#AD443F',  # red
    '#934400', '#BD5700', '#ED6E00', '#FF9030', '#FFA75C',  # orange
    '#936300', '#BD7F00', '#ED9F00', '#FFBB30', '#FFCA5C',  # yellow
    '#285900', '#347201', '#428F01', '#579C1E', '#71AD3F',  # green
    '#005559', '#016D72', '#01898F', '#1E969C', '#3FA8AD',  # blue
    '#072163', '#0C2D7F', '#113A9F', '#3054AD', '#506FBB', ]))  # purple
SHAPES = ['diamond', 'point', 'circle']


def get_caipirinha_config(config, indentation=0):
    limonero_conf = config['juicer']['services']['limonero']
    caipirinha_conf = config['juicer']['services']['caipirinha']
    result = dedent("""
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
    if indentation:
        return '\n'.join(
            ['{}{}'.format(' ' * indentation, r) for r in result.split('\n')])
    else:
        return result


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

    def get_audit_events(self):
        return ['DASHBOARD']

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
            "from juicer.util.dataframe_util import SimpleJsonEncoder as enc",
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
                'data': simplejson.dumps(
                    {vis_model}.get_data(), cls=enc, ignore_nan=True),
                'model': {vis_model}
            }})
            """).format(job_id=self.parameters['job_id'], vis_model=vis_model))

        # Register this new dashboard with Caipirinha
        code_lines.append(get_caipirinha_config(self.config))
        code_lines.append(dedent("""
            caipirinha_service.new_dashboard(config, '{title}', {user},
                {workflow_id}, u'{workflow_name}',
                {job_id}, '{task_id}', visualizations, emit_event)
            """.format(
            title=self.title or '',
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
        self.title = parameters.get(self.TITLE_PARAM, '')
        self.column_names = [c.strip() for c in
                             parameters.get(self.COLUMN_NAMES_PARAM, [])]
        self.orientation = parameters.get(self.ORIENTATION_PARAM, '')
        self.id_attribute = parameters.get(self.ID_ATTR_PARAM, [])
        self.value_attribute = parameters.get(self.VALUE_ATTR_PARAM, [])

        # Visualizations are not cached!
        self.supports_cache = False
        self.output = self.named_outputs.get('visualization',
                                             'vis_task_{}'.format(self.order))

    def get_audit_events(self):
        return [auditing.SAVE_VISUALIZATION]

    def get_model_parameters(self):
        result = {}
        invalid = {'configuration', 'export_notebook',
                'hash', 'transpiler', 'parents',
                'parents_by_port', 'my_ports', 'audit_events',
                'task', 'workflow', 'transpiler_utils'}

        for k, v in list(self.parameters.items()):
            if k not in invalid and not isinstance(v, (set,)):
                result[k] = v
        return result

    def get_output_names(self, sep=','):
        return self.output

    def get_model_name(self):
        NotImplementedError(_("Method generate_code should be implemented "
                              "in {} subclass").format(self.__class__))

    def generate_code(self):
        code_lines = [dedent(
            """
            from juicer.spark.vis_operation import {model}
            from juicer.util.dataframe_util import SimpleJsonEncoder as enc

            params = {params}
            {out} = {model}(
                {input}, '{task}', '{op}',
                '{op_slug}', '{title}',
                {columns},
                '{orientation}', {id_attr}, {value_attr},
                params=params)
            """.format(out=self.output,
                       model=self.get_model_name(),
                       input=self.named_inputs.get('input data', '"None"'),
                       task=self.parameters['task']['id'],
                       op=self.parameters['operation_id'],
                       op_slug=self.parameters['operation_slug'],
                       title=self.title,
                       columns=json.dumps(self.column_names),
                       orientation=self.orientation,
                       id_attr=self.id_attribute,
                       value_attr=self.value_attribute,
                       params=repr(self.get_model_parameters() or {}),
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
                'data': json.dumps({out}.get_data(), cls=enc, ignore_nan=True),
            }}""").format(job_id=self.parameters['job_id'],
                          out=self.output))

            code_lines.append(dedent("""
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
        return BarChartModel.__name__


class PieChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return PieChartModel.__name__


class DonutChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return DonutChartModel.__name__


class LineChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return LineChartModel.__name__


class AreaChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return AreaChartModel.__name__


class TableVisualizationOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return TableVisualizationModel.__name__


class ScatterPlotOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return ScatterPlotModel.__name__

# New visualizations 2020
class IndicatorOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return IndicatorModel.__name__

class MarkdownOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)
        self.has_code = True

    def get_model_name(self):
        return MarkdownModel.__name__

class WordCloudOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return WordCloudModel.__name__


class HeatmapOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return HeatmapModel.__name__

class BubbleChartOperation(VisualizationMethodOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return BubbleChartModel.__name__

class ForceDirectOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return ForceDirectModel.__name__


class IFrameOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)
        self.has_code = True

    def get_model_name(self):
        return IFrameModel.__name__

class TreemapOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return TreemapModel.__name__

#=============================
class MapOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):

        if parameters.get('type') in ['polygon', 'geojson']:
            limonero_config = parameters['configuration']['juicer']['services'][
                'limonero']
            url = limonero_config['url']
            token = str(limonero_config['auth_token'])

            metadata = limonero_service.get_data_source_info(
                url, token, parameters.get('polygon'))
            if not metadata.get('url'):
                raise ValueError(
                    _('Incorrect data source configuration (empty url or '
                      'not GEOJSON)'))
            else:
                parameters['polygon_url'] = metadata['url']
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return MapModel.__name__


class SummaryStatisticsOperation(VisualizationMethodOperation):
    ATTRIBUTES_PARAM = 'attributes'
    CORRELATION_PARAM = 'correlation'

    def __init__(self, parameters, named_inputs, named_outputs):
        if not parameters.get(self.TITLE_PARAM):
            parameters[self.TITLE_PARAM] = 'Summary'
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, None)
        self.correlation = parameters.get(self.CORRELATION_PARAM) \
                           in TRUE_VALS

    def get_model_parameters(self):
        return {self.ATTRIBUTES_PARAM: self.attributes or [],
                self.CORRELATION_PARAM: self.correlation}

    def get_model_name(self):
        return SummaryStatisticsModel.__name__


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
        self.default_time_format = '%Y-%m-%d'

        if len(id_attribute) > 0 and isinstance(id_attribute, list):
            self.id_attribute = id_attribute[0]
        else:
            self.id_attribute = id_attribute

        self.value_attribute = value_attribute

    def get_data(self):
        raise NotImplementedError(_('Should be implemented in derived classes'))

    def get_schema(self):
        return self.data.schema.json()

    def get_icon(self):
        return 'fa-question-o'

    def get_column_names(self):
        return ""


class ChartVisualization(VisualizationModel):
    def get_data(self):
        raise NotImplementedError(_('Should be implemented in derived classes'))

    @staticmethod
    def _get_attr_type(attr):
        if isinstance(attr, Iterable):
            return [ChartVisualization._get_attr_type(a) for a in attr]
        elif attr.dataType.jsonValue() == 'date':
            attr_type = 'date'
        elif attr.dataType.jsonValue() == 'datetime':
            attr_type = 'time'
        elif attr.dataType.jsonValue() == 'time':
            attr_type = 'text'
        elif attr.dataType.jsonValue() == 'timestamp':
            attr_type = 'time'
        elif attr.dataType.jsonValue() == 'text':
            attr_type = 'text'
        elif attr.dataType.jsonValue() == 'character':
            attr_type = 'text'
        else:
            attr_type = 'number'

        return attr_type

    def _get_title_legend_tooltip(self):
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
                    "<span class='metric'>{{x}}</span>"
                    "<span class='number'>{{y}}</span>"
                ]
            },
        }

    def _get_axis_info(self, single_x_attr=True):
        schema = self.data.schema
        if not self.params.get('x_axis_attribute'):
            raise ValueError(_('X-axis attribute not specified'))
        if single_x_attr:
            x = self.params.get('x_axis_attribute')[0]
        else:
            x = self.params.get('x_axis_attributes')

        x_attr = [c for c in schema if c.name == x]
        y_attrs = [c for c in schema if c.name in self.column_names]
        if len(x_attr):
            if single_x_attr:
                x_attr = x_attr[0]
        else:
            raise ValueError(
                _('Attribute {} for X-axis does not exist in ({})').format(
                    x, ', '.join([c.name for c in schema])))
        if len(y_attrs) == 0:
            raise ValueError(_(
                'At least one attribute for Y-axis does not exist: {}').format(
                ', '.join(self.params.get('column_names', []))))

        x_type = ChartVisualization._get_attr_type(x_attr)
        return x_attr, x_type, y_attrs

    @staticmethod
    def _format(value):
        if value is None:
            return None
        elif any([isinstance(value, datetime.datetime),
                  isinstance(value, datetime.date)]):
            return value.isoformat()
        elif isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return value

    def to_pandas(self):
        return self.data.toPandas()

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
            color = COLORS_PALETTE[(i % 6) * 5 + ((i // 6) % 5)]
            colors[attr.name] = {
                'fill': color,
                'gradient': color,
                'stroke': color,
            }
            color_counter = i

        result = {}
        result.update(self._get_title_legend_tooltip())

        # For barcharts this is right option
        result['legend']['text'] = '{{x}}'

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
        elif x_type in ['timestamp', 'date', 'time']:
            # lets have this hardcoded for now
            result['x']["inFormat"] = self.default_time_format
            result['x']["outFormat"] = self.default_time_format

        for inx_row, row in enumerate(rows):
            x_value = row[x_attr.name]
            if x_value not in colors:
                inx_row += 1
                color = COLORS_PALETTE[(color_counter % 6) * 5 +
                                       ((color_counter // 6) % 5)]
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
                    (inx_row % 6) * 5 + ((inx_row // 6) % 5)],
                'values': []
            }
            result['data'].append(data)
            for i, attr in enumerate(y_attrs):
                data['values'].append(
                    {
                        'x': attr.name,
                        'name': LineChartModel._format(x_value),
                        'y': LineChartModel._format(row[attr.name]),
                    }
                )
                if i >= 100:
                    raise ValueError(
                        _('The maximum number of values for x-axis is 100.'))

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
            # Important to use only first item!
            label = self.value_attribute[0]

        value_attr = [c for c in schema if c.name == self.value_attribute[0]]
        if len(value_attr):
            value_attr = value_attr[0]
        else:
            raise ValueError(
                _('Attribute {} does not exist in ({})').format(
                    label, ', '.join([c.name for c in schema])))

        label_attr = [c for c in schema if c.name == label]
        if len(label_attr):
            label_attr = label_attr[0]
        else:
            raise ValueError(
                _('Attribute {} for label does not exist in ({})').format(
                    label, ', '.join([c.name for c in schema])))
        return label_attr, None, value_attr

    def get_data(self):
        label_attr, ignored, value_attr = self._get_axis_info()

        # @FIXME Spark 2.2.0 is raising an exception if self.data.collect()
        # is called directly when the output port is used multiple times.
        self.data.count()
        rows = self.data.collect()
        result = self._get_title_legend_tooltip()
        result['legend']['isVisible'] = self.params.get('legend') in TRUE_VALS

        x_format = self.params.get("x_format", {})
        if not isinstance(x_format, dict):
            x_format = {'key': x_format}

        result.update({
            "x": {
                "title": self.params.get("x_title"),
                "value": "sum",
                "color": "#222",
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
                "format": x_format.get('key'),
            },
            "data": [],
            "pie_type": self.params.get('type', 'pie'),

        })
        for i, row in enumerate(rows):
            data = {
                'x': float(row[value_attr.name]),
                'value': float(row[value_attr.name]),
                'id': '{}_{}'.format(label_attr.name, i),
                'name': row[label_attr.name],
                'label': row[label_attr.name],
                'color': COLORS_PALETTE[(i % 6) * 5 + ((i // 6) % 5)],
            }
            result['data'].append(data)
            if i >= 100:
                raise ValueError(
                    _('The maximum number of values for this chart is 100.'))
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
                "color": COLORS_PALETTE[(i % 6) * 5 + ((i // 6) % 5)],
                "pointColor": COLORS_PALETTE[(i % 6) * 5 + ((i // 6) % 5)],
                "pointShape": SHAPES[i % len(SHAPES)],
                "pointSize": 3,
                "values": []
            })

        result = {}
        result.update(self._get_title_legend_tooltip())

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
            "data": data,
            'using_date': x_type in ('date', 'time')
        })

        if x_type in ['number']:
            result['x']['format'] = self.params.get("x_format", {}).get('key')
        elif x_type == 'time':
            # FIXME: gViz does not handles datetime correctly
            result['x']['inFormat'] = '%Y-%m-%dT%H:%M:%S'
            result['x']['outFormat'] = '%Y-%m-%d'
        elif x_type in ['date']:
            result['x']["inFormat"] = self.default_time_format
            result['x']["outFormat"] = self.default_time_format
            result['x']["type"] = 'time'  # FIXME

        for row in rows:
            for i, attr in enumerate(y_attrs):
                data[i]['values'].append(
                    {
                        "x": LineChartModel._format(row[x_attr.name]),
                        "y": LineChartModel._format(row[attr.name]),
                    }
                )
        return result


class MapModel(ChartVisualization):
    def get_icon(self):
        return 'fa-map-marker'

    def get_data(self):
        result = {}
        result.update(self._get_title_legend_tooltip())
        rows = self.data.collect()

        if self.params.get('value'):
            value_attr = next((c for c in self.data.schema if
                               c.name == self.params['value'][0]), None)

        param_map_type = self.params.get('type', 'heatmap')

        map_type = {
            'heatmap': 'heatmap',
            'points': 'points',
            'polygon': 'polygon'
        }[param_map_type]

        result['mode'] = {
            map_type: True
        }
        if param_map_type == 'polygon':
            result['geojson'] = {
                'url': self.params.get('polygon_url'),
                'idProperty': self.params.get('geojson_id', 'id') or 'id'
            }

        data = []
        result['data'] = data

        lat = self.params.get('latitude', [None])[0]
        lng = self.params.get('longitude', [None])[0]
        label = self.params.get('label', [None])[0]

        for i, row in enumerate(rows):
            if self.params.get('value'):
                value = row[self.params.get('value')[0]]
            else:
                value = 0
            if param_map_type == 'polygon':

                info = {"id": row[label], "value": value}
                extra = self.params.get('extra_data', [])
                for f in extra:
                    if f in row:
                        info[f] = row[f]

            else:
                info = {
                    "id": str(i), "value": value,
                    "name": row[label] if label else None,
                }
                if lat and lng:
                    info["lat"] = row[lat]
                    info["lon"] = row[lng]

            data.append(info)

        return result


class AreaChartModel(LineChartModel):
    def get_icon(self):
        return 'fa-area-chart'


class ScatterPlotModel(ChartVisualization):
    """
    Scatter plot chart model
    """

    # noinspection PyArgumentEqualDefault
    def get_data(self):
        schema = self.data.schema

        result = {}
        attrs = {}
        for axis in ['x', 'y', 'z', 't']:
            name = self.params.get('{}_axis_attribute'.format(axis), [None])
            if isinstance(name, list) and len(name):
                name = name[0]
            else:
                name = None
            attrs[axis] = next((c for c in schema if c.name == name), None)
            if attrs[axis]:
                axis_type = ChartVisualization._get_attr_type(attrs[axis])

                # this way we don't bind x_axis and y_axis types. Y is only
                # going to be number for now
                if axis == 'y':
                    axis_type = 'number'

                result[axis] = {
                    "title": self.params.get("{}_title".format(axis)),
                    "prefix": self.params.get("{}_prefix".format(axis)),
                    "suffix": self.params.get("{}_suffix".format(axis)),
                    "type": axis_type
                }
                axis_format = self.params.get('{}_format'.format(axis), {})

                if axis_type in ['number']:
                    result[axis]['format'] = axis_format.get('key')

                elif axis_type in ['timestamp', 'date', 'time']:
                    result[axis]["inFormat"] = self.default_time_format
                    result[axis]["outFormat"] = self.default_time_format

                    # result[axis]["outFormat"] = axis_format.get('key')
                    # result[axis]["inFormat"] = axis_format.get('key')

        result.update(self._get_title_legend_tooltip())

        series_attr_name = self.params.get('series_attribute', [None])[0]
        if series_attr_name:
            series_attr = next(
                (c for c in schema if c.name == series_attr_name), None)
        else:
            series_attr = None

        series = {}
        series_key = '@_ \\UNIQUE KEY/ :P_ @'
        if not series_attr:
            series[series_key] = {
                "id": result['title'],
                "name": result['title'],
                "image": None,
                "color": COLORS_PALETTE[0],
                "values": []
            }

        rows = self.data.collect()
        current_color = 0
        for row in rows:
            if series_attr:
                series_value = row[series_attr.name]
                if series_value not in series:
                    color = COLORS_PALETTE[(current_color % 6) * 5 +
                                           ((current_color // 6) % 5)]
                    series[series_value] = {
                        "id": series_value,
                        "name": series_value,
                        "image": None,
                        "color": color,
                        "values": []
                    }
                    current_color += 1
                data = series[series_value]['values']
            else:
                data = series[series_key]['values']

            item = {}
            for axis in ['x', 'y', 'z', 't']:
                item[axis] = ScatterPlotModel._get_value(row, attrs[axis])
            data.append(item)

        result['data'] = list(series.values())
        return result

    @staticmethod
    def _get_value(row, attr, default_value=None):
        if attr is not None:
            return ChartVisualization._format(row[attr.name])
        else:
            return default_value


class HtmlVisualizationModel(VisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data=None, task_id=None, type_id=1, type_name=None,
                 title=None,
                 column_names=None,
                 orientation=None, id_attribute=None,
                 value_attribute=None, params=None):
        type_id = 1
        type_name = 'html'
        if id_attribute is None:
            id_attribute = []
        if value_attribute is None:
            value_attribute = []
        if column_names is None:
            column_names = []
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
            rows = self.data.limit(500).select(*self.column_names).collect()
        else:
            rows = self.data.limit(500).collect()

        return {"rows": rows,
                "attributes": self.get_column_names().split(',')}

    def get_schema(self):
        if self.column_names:
            return self.data.select(*self.column_names).schema.json()
        else:
            return self.data.schema.json()

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
        self.names = [_('attribute'), _('max'), _('min'), _('std. dev.'),
                      _('count'), _('avg'),
                      _('approx. distinct'), _('missing'), _('skewness'),
                      _('kurtosis')]

        if self.params['correlation']:
            self.names.extend(
                [_('correlation to {} (Pearson)').format(attr) for attr in
                 self.attrs])
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

            if name in self.numeric_attrs:
                stats.append(
                    functions.round(functions.skewness(df_col), 2).alias(
                        'skewness_{}'.format(name)))
                stats.append(
                    functions.round(functions.kurtosis(df_col), 2).alias(
                        'kurtosis_{}'.format(name)))
            else:
                stats.append(functions.lit('-'))
                stats.append(functions.lit('-'))

            if self.params['correlation']:
                for pair in corr_pairs[i]:
                    if all([pair[0] in self.numeric_attrs,
                            pair[1] in self.numeric_attrs]):
                        stats.append(
                            functions.round(functions.corr(*pair), 4).alias(
                                'corr_{}'.format(i)))
                    else:
                        stats.append(functions.lit('-'))

        self.data = self.data.agg(*stats)
        aggregated = self.data.take(1)[0]
        n = len(self.names)
        rows = [aggregated[i:i + n] for i in range(0, len(aggregated), n)]

        return {"rows": rows, "attributes": self.get_column_names().split(',')}


class BoxPlotOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return BoxPlotModel.__name__


BoxPlotInfo = collections.namedtuple(
    'BoxPlotInfo', ['fact', 'q1', 'q2', 'q3', 'outliers', 'min', 'max'])


class BoxPlotModel(ChartVisualization):
    """ Box plot model for visualization of data """

    def get_icon(self):
        return 'fa-chart'

    # noinspection PyUnresolvedReferences
    def get_data(self):
        from pyspark.sql import functions as fns

        def _alias(attribute, counter, suffix):
            return '{}_{}_{}'.format(attribute, counter, suffix)

        self.data.cache()

        facts = self.params.get('fact_attributes')

        if facts is None or not isinstance(facts, list) or len(facts) == 0:
            raise ValueError(
                _('Input attribute(s) must be informed for box plot.'))

        quartiles_expr = [
            fns.expr('percentile_approx({}, array(.25, .5, .75))'.format(fact)
                     ).alias(fact) for fact in facts]

        group = self.params.get('group_attribute')
        # Calculates the quartiles for fact attributes
        if group is not None and len(group) >= 1:
            group = group[0]
            quartiles = self.data.groupBy(group).agg(
                *quartiles_expr).collect()
        else:
            group = None
            quartiles = self.data.agg(*quartiles_expr).collect()

        computed_cols = []

        show_outliers = self.params.get('show_outliers') in TRUE_VALS
        group_offset = 1 if group is not None else 0
        for i, quartile_row in enumerate(quartiles):
            # First column in row is the label for the group, so it's ignored
            for j, fact_quartiles in enumerate(quartile_row[group_offset:]):
                # Calculates inter quartile range (IQR)
                iqr = round(fact_quartiles[2] - fact_quartiles[0], 4)

                # Calculates boundaries for identifying outliers
                lower_bound = round(float(fact_quartiles[0]) -
                        1.5 * float(iqr), 4)
                upper_bound = round(float(fact_quartiles[2]) +
                        1.5 * float(iqr), 4)

                # Outliers are beyond boundaries
                outliers_cond = (fns.col(facts[j]) < fns.lit(lower_bound)) | (
                    fns.col(facts[j]) > fns.lit(upper_bound))

                # If grouping is not specified, uses True when combining
                # conditions
                if group is not None:
                    value_cond = fns.col(group) == fns.lit(quartile_row[0])
                else:
                    value_cond = fns.lit(True)

                if show_outliers:
                    outliers = fns.collect_list(
                        fns.when(outliers_cond & value_cond,
                                 fns.col(facts[j]))).alias(
                        _alias(facts[j], i, 'outliers'))
                else:
                    outliers = fns.lit(None).alias(
                        _alias(facts[j], i, 'outliers'))
                computed_cols.append(outliers)

                min_val = fns.min(
                    fns.when(~outliers_cond & value_cond,
                             fns.col(facts[j]))).alias(
                    _alias(facts[j], i, 'min'))
                computed_cols.append(min_val)
                max_val = fns.max(
                    fns.when(~outliers_cond & value_cond,
                             fns.col(facts[j]))).alias(
                    _alias(facts[j], i, 'max'))
                computed_cols.append(max_val)
        if group is not None:
            min_max_outliers = self.data.groupBy(group).agg(
                *computed_cols).collect()
        else:
            min_max_outliers = self.data.agg(*computed_cols).collect()

        # Organize all information
        summary = {}
        for i, quartile_row in enumerate(quartiles):
            summary_row = []
            if group:
                summary[quartile_row[0]] = summary_row
            else:
                summary[i] = summary_row

            # Data for min, max and outliers are organized in multiple columns,
            # like a matrix represented as a vector. This offset is used to
            # re-organize the data
            offset = group_offset + 3 * len(facts) * i
            for j, fact in enumerate(facts):
                start_offset = offset + j * 3  # len(facts)
                end_offset = offset + (1 + j) * 3  # len(facts)
                min_max_outliers_part = min_max_outliers[i][
                                        start_offset: end_offset]

                box_plot_info = BoxPlotInfo(fact=fact,
                                            q1=quartile_row[j + group_offset][
                                                0],
                                            q2=quartile_row[j + group_offset][
                                                1],
                                            q3=quartile_row[j + group_offset][
                                                2],
                                            outliers=min_max_outliers_part[0],
                                            min=min_max_outliers_part[1],
                                            max=min_max_outliers_part[2])
                summary_row.append(box_plot_info)
        # Format to JSON
        result = []
        v = {
            'chart': {'type': 'boxplot'},
            'title': {'text': self.params.get('title', '')},
            'legend': {'enabled': False},
            'xAxis': {
                'categories': [],
                'title': {'text': self.params.get('x_title')}
            },
            'yAxis': {
                'title': {'text': self.params.get('y_title')},
            },
            'series': [
                {
                    'name': self.params.get('y_title'),
                    'data': [],
                    'tooltip': {'headerFormat': '<b>{point.key}</b><br/>'}
                },
                {
                    'name': 'Outlier',
                    'type': 'scatter',
                    'data': [],
                    'marker': {
                        'fillColor': 'white',
                        'lineWidth': 1
                    },
                    'tooltip': {
                        'pointFormat': '{point.y}'
                    }
                }
            ]
        }
        result.append(v)
        for i, (k, rows) in enumerate(summary.items()):
            for j, s in enumerate(rows):
                if group:
                    v['xAxis']['categories'].append(
                        '{}: {}'.format(k, facts[j]))
                else:
                    v['xAxis']['categories'].append(facts[j])
                v['series'][0]['data'].append([s.min, s.q1, s.q2, s.q3, s.max])
                if s.outliers:
                    v['series'][1]['data'].extend([[j, o] for o in s.outliers])

        if not show_outliers:
            del v['series'][1]
        return {'data': result}


class HistogramOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return HistogramModel.__name__


class HistogramModel(ChartVisualization):
    """ Histogram model for visualization of data """

    def get_icon(self):
        return 'fa-chart'

    # noinspection PyUnresolvedReferences
    def get_data(self):
        from pyspark.sql import functions

        self.data.cache()

        attributes = self.params.get('attributes')
        bins = int(self.params.get('bins', '10'))

        if attributes is None or not isinstance(attributes, list) or len(
                attributes) == 0:
            raise ValueError(
                _('Input attribute(s) must be informed for histogram.'))
        schema = dict((a.name, a) for a in self.data.schema)
        cols = []
        numeric_types = ['integer', 'int', 'float', "byte", "long", "short"]
        for attribute in attributes:
            if attribute in schema:
                type_name = schema[attribute].dataType.typeName()
                if type_name == 'decimal':
                    cols.append(functions.col(attribute).cast('float'))
                elif type_name in numeric_types:
                    cols.append(functions.col(attribute))
                else:
                    raise ValueError(
                        _('Attribute {} must be numeric (actual: {}).').format(
                            attribute, type_name))
            else:
                raise ValueError(
                    _('Attribute {} not found in input data.').format(
                        attribute))

        def identity(x):
            return x

        hist_data = []
        # For each attribute, it has to read input once
        for i, col in enumerate(cols):
            # data contains a vector with 2 elements:
            # the first one, with the ranges' boundaries and the 2nd with
            # frequencies.
            data = self.data.select(col).rdd.flatMap(identity).histogram(bins)
            v = {
                'chart': {'type': 'column'},
                'title': {'text': self.params.get('title', '')},
                'xAxis': {
                    'tickWidth': 1,
                    'tickmarkPlacement': 'between',
                    'title': {'text': '{} {}'.format(
                        self.params.get('x_title'), attributes[i])
                    },
                    'categories': [round(boundary, 4) for boundary in data[0]],
                },
                'yAxis': {'title': {'text': self.params.get('y_title')}},
                'plotOptions': {
                    'column': {
                        'pointPadding': 0,
                        'borderWidth': 1,
                        'groupPadding': 0,
                        'shadow': False,
                        'pointPlacement': 'between',
                    }
                },
                'series': [{
                    'name': _('Histogram'),
                    'data': data[1]

                }]
            }
            hist_data.append(v)
        return {'data': hist_data}

# Models 2020

class IndicatorModel(ChartVisualization):
    """ Indicator (odometer/big number)  model for data visualization. """

    def get_data(self):

        data = self.to_pandas()

        displays =[]
        number = True
        delta = False

        if self.params.get('display_value', False) in TRUE_VALS:
            displays.append('number')
        if self.params.get('display_delta', False) in TRUE_VAL:
            displays.append('delta')
            delta = True
        if self.params.get('display_gauge', False) in TRUE_VALS:
            displays.append('gauge')

        mode = '+'.join(displays)
        result = {
            'type': 'indicator',
            'title': self.params.get('title')[0],
            'mode': mode or 'number',
            'footer': self.params.get('footer')
        }
        if delta:
           delta_attr = self.params.get('delta', []) or []
           if len(delta_attr) == 0:
                raise ValueError(
                    _('Parameter delta must be informed if '
                       'display delta option is enabled.'))
           result['delta'] = {'reference': data[delta_attr[0]].iloc[0]}
           if self.params.get('delta_relative') in TRUE_VALS:
               result['delta']['relative'] = True

        if number:
           value_attr = self.params.get('value', []) or []
           if len(value_attr) == 0:
                raise ValueError(
                    _('Parameter value must be informed if '
                       'display value option is enabled.'))
           result['value'] = data[value_attr[0]].iloc[0]

        return {'data': result}

class IFrameModel(ChartVisualization):
    """ IFrame visualization. """

    def get_data(self):
        link = self.params.get('link')
        if link is None:
           raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        'link', 'Iframe'))
        return {'data': {'link': link}}

class MarkdownModel(ChartVisualization):
    """ Markdown visualization. """

    def get_data(self):
        text = self.params.get('text')
        if text is None:
           raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        'text', 'Markdown'))
        return {'data': {'text': text}}

class BubbleChartModel(ChartVisualization):
    """ Bubble chart visualization chart """

    TITLE_PARAM = 'title'
    X_AXIS_ATTRIBUTE_PARAM = 'x_axis_attribute'
    Y_AXIS_ATTRIBUTE_PARAM = 'y_axis_attribute'
    SIZE_ATTRIBUTE_PARAM = 'size_attribute'
    COLOR_ATTRIBUTE_PARAM = 'color_attribute'
    TEXT_ATTRIBUTE_PARAM = 'text_attribute'
    X_TITLE_PARAM = 'x_title'
    Y_TITLE_PARAM = 'y_title'
    COLOR_PALETTE_PARAM = 'color_palette'

    def get_data(self):
        data = self.to_pandas()

        x_name = self.params.get(self.X_AXIS_ATTRIBUTE_PARAM)
        if x_name is None or len(x_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.X_AXIS_ATTRIBUTE_PARAM, 'Bubble chart'))
        else:
            x_name = x_name[0]

        y_name = self.params.get(self.Y_AXIS_ATTRIBUTE_PARAM)
        if y_name is None or len(y_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.Y_AXIS_ATTRIBUTE_PARAM, 'Bubble chart'))
        else:
            y_name = y_name[0]

        size_name = self.params.get(self.SIZE_ATTRIBUTE_PARAM)
        if size_name is None or len(size_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.SIZE_ATTRIBUTE_PARAM, 'Bubble chart'))
        else:
            size_name = size_name[0]

        color_name = self.params.get(self.COLOR_ATTRIBUTE_PARAM)
        if color_name and len(color_name) > 0:
            color_name = color_name[0]
        else:
            color_name = None

        text_name = self.params.get(self.TEXT_ATTRIBUTE_PARAM)
        if text_name and len(text_name) > 0:
            text_name = text_name[0]
        else:
            text_name = None

        x, y, s, c, t = [], [], [], [], []

        palette = self.params.get(self.COLOR_PALETTE_PARAM, COLORS_PALETTE)
        color_map = {}

        for index, row in data.iterrows():
            x.append(row[x_name])
            y.append(row[y_name])
            s.append(row[size_name])
            if color_name:
                color_value = row[color_name]
                if color_value not in color_map:
                    color_map[color_value] = palette[len(color_map) % len(palette)]
                c.append(color_map[color_value])
            if text_name:
                t.append(row[text_name])

        result = {
            'type': 'scatter',
            'mode': 'markers',
            'title': self.params.get('title'),
            'x': x,
            'y': y,
            'sizes': s,
            'colors': c,
            'texts': t,
            'x_title': self.params.get(self.X_TITLE_PARAM),
            'y_title': self.params.get(self.Y_TITLE_PARAM),
        }
        return {'data': result}

class HeatmapModel(ChartVisualization):
    """ Heatmap visualization """

    TITLE_PARAM = 'title'
    ROW_ATTRIBUTE_PARAM = 'row_attribute'
    COLUMN_ATTRIBUTE_PARAM = 'column_attribute'
    VALUE_ATTRIBUTE_PARAM = 'value_attribute'
    AGGREGATION_FUNCTION_PARAM = 'aggregation_function'
    ROW_TITLE_PARAM = 'row_title'
    COLUMN_TITLE_PARAM = 'column_title'
    COLOR_SCALE_PARAM = 'color_scale'

    def _compute(self, agg_function, group, attr):

        from pyspark.sql import functions
        if agg_function == 'count':
            df = self.data.groupBy(*group).count().withColumnRenamed(
                    'count', 'tmp_value')
        else:
            f = getattr(functions, agg_function)
            df = self.data.groupBy(*group).agg(
                    f(functions.lit(attr)).alias('tmp_value'))

        return df.toPandas()

    def get_data(self):

        col_name = self.params.get(self.COLUMN_ATTRIBUTE_PARAM)
        if col_name is None or len(col_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.COLUMN_ATTRIBUTE_PARAM, 'Heatmap'))
        else:
            col_name = col_name[0]

        row_name = self.params.get(self.ROW_ATTRIBUTE_PARAM)
        if row_name is None or len(row_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.ROW_ATTRIBUTE_PARAM, 'Heatmap'))
        else:
            row_name = row_name[0]

        value_name = self.params.get(self.VALUE_ATTRIBUTE_PARAM)
        if value_name is None or len(value_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.VALUE_ATTRIBUTE_PARAM, 'Heatmap'))
        else:
            value_name = value_name[0]

        rows, cols, values = [], [], []

        palette = self.params.get(self.COLOR_SCALE_PARAM, COLORS_PALETTE)
        agg_function = self.params.get(self.AGGREGATION_FUNCTION_PARAM, 'sum')

        data = self._compute(agg_function, [row_name, col_name], value_name)
        color_map = {}

        for index, row in data.iterrows():
            cols.append(row[col_name])
            rows.append(row[row_name])
            values.append(row['tmp_value'])

        result = {
            'type': 'scatter',
            'mode': 'markers',
            'title': self.params.get('title'),
            'rows': rows,
            'cols': cols,
            'values': values,
            'colors': self.params.get(self.COLOR_SCALE_PARAM, COLORS_PALETTE),
            'row_title': self.params.get(self.ROW_TITLE_PARAM),
            'column_title': self.params.get(self.COLUMN_TITLE_PARAM),
        }
        return {'data': result}

class TreemapModel(ChartVisualization):
    """ Treemap visualization """

    TITLE_PARAM = 'title'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    PARENT_ATTRIBUTE_PARAM = 'parent_attribute'
    VALUE_ATTRIBUTE_PARAM = 'value_attribute'
    DISPLAY_VALUE_PARAM = 'display_value'
    DISPLAY_ENTRY_PERCENTAGE_PARAM = 'display_entry_percentage'
    DISPLAY_PARENT_PERCENTAGE_PARAM = 'display_parent_percentage'

    COLOR_PALETTE_PARAM = 'color_palette'

    def get_data(self):

        label = self.params.get(self.LABEL_ATTRIBUTE_PARAM)
        if label is None or len(label) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.LABEL_ATTRIBUTE_PARAM, 'Treemap'))
        else:
            label = label[0]

        parent = self.params.get(self.PARENT_ATTRIBUTE_PARAM)
        if parent is None or len(parent) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.PARENT_ATTRIBUTE_PARAM, 'Treemap'))
        else:
            parent = parent[0]

        value_name = self.params.get(self.VALUE_ATTRIBUTE_PARAM)
        if value_name is None or len(value_name) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.VALUE_ATTRIBUTE_PARAM, 'Treemap'))
        else:
            value_name = value_name[0]


        data = self.data.toPandas()
        labels, parents, values, colors = [], [], [], []
        
        palette = self.params.get(self.COLOR_PALETTE_PARAM)

        for index, row in data.iterrows():
            labels.append(row[label])
            parents.append(row[parent])
            values.append(row[value_name])
            if palette:
                colors.append(palette[index % len(palette)])

        textinfo = ['label']

        if self.params.get(self.DISPLAY_VALUE_PARAM) in TRUE_VALS:
            textinfo.append('value')
        if self.params.get(self.DISPLAY_ENTRY_PERCENTAGE_PARAM) in TRUE_VALS:
            textinfo.append('percent entry')
        if self.params.get(self.DISPLAY_PARENT_PERCENTAGE_PARAM) in TRUE_VALS:
            textinfo.append('percent parent')

        result = {
            'branchvalues': 'total',     
            'type': 'treemap',
            'labels': labels,
            'parents': parents,
            'values': values,
            'title': self.params.get('title'),
            'colors': colors,
            'textinfo': '+'.join(textinfo),
        }
        return {'data': result}


