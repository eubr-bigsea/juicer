# -*- coding: utf-8 -*-
import itertools
import json
from textwrap import dedent
import datetime

from juicer.operation import Operation
from juicer.util import chunks
from juicer.util import dataframe_util
from juicer.util.dataframe_util import get_csv_schema


class PublishVisOperation(Operation):
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
                'model': {vis_model}
            }})
            """).format(job_id=self.parameters['job_id'], vis_model=vis_model))

        limonero_conf = self.config['juicer']['services']['limonero']
        caipirinha_conf = self.config['juicer']['services']['caipirinha']

        # Register this new dashboard with Caipirinha
        code_lines.append(dedent("""
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
            }}
            caipirinha_service.new_dashboard(config, '{title}', {user},
                {workflow_id}, '{workflow_name}',
                {job_id}, '{task_id}', visualizations, emit_event)""".format(
            limonero_url=limonero_conf['url'],
            limonero_token=limonero_conf['auth_token'],
            caipirinha_url=caipirinha_conf['url'],
            caipirinha_token=caipirinha_conf['auth_token'],
            storage_id=caipirinha_conf['storage_id'],
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
        self.title = parameters.get(self.TITLE_PARAM, '')
        self.column_names = parameters.get(self.COLUMN_NAMES_PARAM, '')
        self.orientation = parameters.get(self.ORIENTATION_PARAM, '')
        self.id_attribute = parameters.get(self.ID_ATTR_PARAM, [])
        self.value_attribute = parameters.get(self.VALUE_ATTR_PARAM, [])

        # Visualizations are not cached!
        self.supports_cache = False
        self.output = self.named_outputs.get('visualization',
                                             'vis_task_'.format(self.order))

    def get_model_parameters(self):
        return {}

    def get_output_names(self, sep=','):
        return self.output

    def get_model_name(self):
        NotImplementedError("Method generate_code should be implemented "
                            "in {} subclass".format(self.__class__))

    def generate_code(self):
        code = \
            """
            from juicer.spark.vis_operation import {model}
            {out} = {model}(
                {input}, '{task}', '{op}',
                '{op_slug}', '{title}',
                '{columns}',
                '{orientation}',{id_attr},{value_attr},
                params={params})
            """.format(out=self.output,
                       model=self.get_model_name(),
                       input=self.named_inputs['input data'],
                       task=self.parameters['task']['id'],
                       op=self.parameters['operation_id'],
                       op_slug=self.parameters['operation_slug'],
                       title=self.title,
                       columns=self.column_names,
                       orientation=self.orientation,
                       id_attr=self.id_attribute,
                       value_attr=self.value_attribute,
                       params=json.dumps(self.get_model_parameters()))
        return dedent(code)


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


class TableVisOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return 'TableVisualizationModel'


class ScatterPlotOperation(VisualizationMethodOperation):
    ATTRIBUTES_PARAM = 'attributes'
    CALCULATE_PARAM = 'calculate'

    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        self.calculate = parameters.get(self.CALCULATE_PARAM)

    def get_model_parameters(self):
        return {self.ATTRIBUTES_PARAM: self.attributes,
                self.CALCULATE_PARAM: self.calculate}

    def get_model_name(self):
        return 'ScatterPlotModel'


class SummaryStatisticsOperation(VisualizationMethodOperation):
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
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

class VisualizationModel:
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
        # return data.rdd.map(dataframe_util.convert_to_csv).collect()
        # return data.rdd.map(dataframe_util.convert_to_python).collect()
        raise NotImplementedError('Should be implemented in derived classes')

    # def get_schema(self, data):
    #     return dataframe_util.get_csv_schema(data)
    #
    # def get_dict_schema(self, data):
    #     return dataframe_util.get_dict_schema(data)

    def get_schema(self):
        return self.data.schema.json()

    def get_icon(self):
        return 'fa-question-o'

    def get_column_names(self):
        return ""


class BarChartModel(VisualizationModel):
    def get_data2(self, data):
        rows = data.collect()
        result = []
        columns = [c.strip() for c in self.column_names.split(',')]
        for row in rows:
            values = []
            for i, col in enumerate(self.value_attribute):
                values.append(dict(
                    id=col,
                    name=columns[i],
                    value=row[col]
                ))
            result.append(dict(
                id=row[self.id_attribute],
                name=row[self.id_attribute],
                values=values
            ))
        return result

    def get_icon(self):
        return 'fa-bar-chart'

    def get_data(self):
        """
        Returns data as a list dictionaries in Python (JSON encoder friendly).
        """
        return self.data.rdd.map(
            dataframe_util.format_row_for_visualization).collect()


class PieChartModel(VisualizationModel):
    def get_icon(self):
        return 'fa-pie-chart'

    def get_data(self):
        """
        Returns data as a list dictionaries in Python (JSON encoder friendly).
        """
        return self.data.rdd.map(
            dataframe_util.format_row_for_visualization).collect()


class AreaChartModel(VisualizationModel):
    def get_icon(self):
        return 'fa-area-chart'

    def get_data(self):
        rows = self.data.collect()
        result = []
        columns = [c.strip() for c in self.column_names.split(',')]
        for row in rows:
            values = []
            for i, col in enumerate(self.value_attribute):
                values.append(dict(
                    id=col,
                    name=columns[i],
                    value=row[col]
                ))
            result.append(dict(
                id=row[self.id_attribute],
                name=row[self.id_attribute],
                values=values
            ))
        return result


class LineChartModel(VisualizationModel):
    def get_icon(self):
        return 'fa-line-chart'

    def get_data(self):
        date_types = [datetime.datetime, datetime.date]
        rows = self.data.collect()
        result = []
        columns = [c.strip() for c in self.column_names.split(',')]
        for row in rows:
            values = []
            for i, col in enumerate(self.value_attribute):
                values.append(dict(
                    id=col,
                    name=columns[i],
                    value=row[col] if type(row[col]) not in date_types else row[
                        col].isoformat()
                ))
            result.append(dict(
                id=row[self.id_attribute],
                name=row[self.id_attribute],
                values=values
            ))
        return result


class HtmlVisualizationModel(VisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation, id_attribute, value_attribute, params):
        type_id = 1
        type_name = 'html'
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
    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation, id_attribute, value_attribute, params):
        type_id = 35
        type_name = 'table-visualization'
        VisualizationModel.__init__(self, data, task_id, type_id, type_name,
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return 'fa-table'

    def get_data(self):
        """
        Returns data as tabular (list of lists in Python).
        """
        return self.data.limit(50).rdd.map(
            dataframe_util.convert_to_python).collect()

    def get_column_names(self):
        return self.column_names or get_csv_schema(self.data, only_name=True)


class SummaryStatisticsModel(TableVisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation, id_attribute, value_attribute, params):
        TableVisualizationModel.__init__(self, data, task_id, type_id,
                                         type_name,
                                         title, column_names, orientation,
                                         id_attribute, value_attribute, params)
        self.names = ''
        if len(self.params['attributes']) == 0:
            self.attrs = self.data.schema
        else:
            self.attrs = [attr for attr in self.data.schema if
                          attr.name in self.params['attributes']]
        self.names = ['attribute', 'max', 'min', 'stddev', 'count', 'avg',
                      'approx. distinct', 'missing']

        self.names.extend(
            ['correlation to {}'.format(attr.name) for attr in self.attrs])

        self.column_names = ', '.join(self.names)

    def get_icon(self):
        return 'fa-table'

    def get_data(self):
        """
        Returns statistics about attributes in a data frame
        """

        from pyspark.sql import functions

        # Correlation pairs
        attr_names = [attr.name for attr in self.attrs]
        corr_pairs = list(
            chunks(list(itertools.product(attr_names, attr_names)),
                   len(attr_names)))

        # Cache data
        self.data.cache()

        df_count = self.data.count()

        # TODO: Implement median using df.approxQuantile('col', [.5], .25)

        stats = []
        for i, name in enumerate(attr_names):
            df_col = functions.col(name)
            stats.append(functions.lit(name))
            stats.append(functions.max(df_col).alias('max_{}'.format(name)))
            stats.append(functions.min(df_col).alias('min_{}'.format(name)))
            stats.append(functions.round(
                functions.stddev(df_col), 4).alias('stddev_{}'.format(name)))
            stats.append(functions.count(df_col).alias('count_{}'.format(name)))
            stats.append(functions.round(
                functions.avg(df_col), 4).alias('avg_{}'.format(name)))
            stats.append(functions.approx_count_distinct(df_col).alias(
                'distinct_{}'.format(name)))
            stats.append((df_count - functions.count(df_col)).alias(
                'missing_{}'.format(name)))
            for pair in corr_pairs[i]:
                stats.append(
                    functions.round(functions.corr(*pair), 4).alias(
                        'corr_{}'.format(i)))

        aggregated = self.data.agg(*stats).take(1)[0]
        n = len(self.names)
        return [aggregated[i:i + n] for i in range(0, len(aggregated), n)]


class ScatterPlotModel(VisualizationModel):
    """
    Scatter plot model for visualization.
    Supports calculate covariance or correlation.

    TODO: Implement support to linear regression
    """
    COVARIANCE = 'covariance'
    CORRELATION = 'correlation'

    def get_icon(self):
        return 'fa-table'

    def get_data(self):
        """
        Returns data as tabular (list of lists in Python).
        """
        from pyspark.sql.functions import col, corr, covar_pop
        numerics = [attr.name for attr in self.data.schema if
                    attr.dataType.typeName() in ['int', 'byte', 'long', 'short',
                                                 'float', 'double', 'decimal']]
        attributes = [name for name in self.params['attributes'] if
                      name in numerics]
        calculate = self.params.get('calculate')

        pairs = itertools.combinations(attributes, 2)
        result = {}

        if calculate is not None:
            f = corr if calculate == self.CORRELATION else covar_pop
            aggrs = []
            for pair in pairs:
                aggr_name = '{}_{}_{}'.format(calculate[:3], pair[0], pair[1])
                aggrs = f(col(pair[0]), col(pair[1])).alias(aggr_name)

            calc = self.data.agg(*aggrs).collect()[0].asDict()
            result[calculate] = calc
        result['series'] = []
        for i, pair in enumerate(pairs):
            result['series'][i] = {'attributes': pair, 'data': []}

        for row in self.data.toLocalIterator():
            for i, pair in enumerate(pairs):
                result['series'][i]['data'].append([row[pair[0]], row[pair[1]]])

        return result
