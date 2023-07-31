# coding=utf-8
import copy
from dataclasses import dataclass
import logging.config
from gettext import gettext
from typing import Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.dataframe import Column as SparkCol
from pyspark.sql.dataframe import DataFrame as SparkDF

from juicer.util.dataframe_util import is_numeric_col

from juicer.jobs import get_data_source, get_schema

logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.jobs.analyse_data_source')


@dataclass
class ColumnToSumarize:
    """ Class to store data about column to be summarized """
    spark_column: SparkCol
    name: str
    functions: List[str]
    results: Dict
    is_numeric: bool


def _box_plot(df: SparkDF, col: SparkCol, analysis: Dict) -> Dict[str, any]:

    fig = go.Figure()
    fig.add_trace(_get_box_plot(df, col))
    fig.update_layout(width=500, height=200, autosize=True, margin=dict(
        l=20, r=20, b=30, t=30, pad=4
    ))
    plotly_dict = fig.to_dict()
    del plotly_dict['layout']['template']
    analysis['result'] = plotly_dict

    fig.write_image('lixo2.png')


def _get_box_plot(df: SparkDF, col: SparkCol) -> any:
    """
    Create a box plot for a col in dataframe df.
    """
    # Your data for histogram and box plot
    stat_data = df.approxQuantile(col._jc.toString(), [0.25, 0.5, 0.75], 0.01)
    q1, q2, q3 = stat_data
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = [x[0] for x in df.select(col).filter(
        (col > upper_fence) | (col < lower_fence))
        .toPandas().to_numpy().tolist()]
    return go.Box(
        x=[outliers], q1=[q1], median=[q2], q3=[q2+iqr],
        lowerfence=[lower_fence], upperfence=[upper_fence], orientation='h',)


def _histogram(df: SparkDF, col: SparkCol, name: str, results: Dict,
               analysis: Dict, total_count: int, i: int, is_numeric: bool,
               version: tuple) -> Dict[str, any]:
    # Compute the histogram
    bin_edges, bin_counts = (
        df.select(col).rdd.flatMap(lambda x: x).histogram(20))

    # Format the x-axis labels as "[start, end]"
    x_axis_labels = [f"[{start:.2f}, {end:.2f}]" for start, end
                     in zip(bin_edges[:-1], bin_edges[1:])]

    # Create subplots with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01, row_heights=[0.9, 0.1])

    # Add histogram trace to the first subplot
    fig.add_trace(
        go.Bar(x=x_axis_labels, y=bin_counts,
               # title='Histogram of Sepal Width',
               #labels={'x': 'Sepal Width', 'y': 'Frequency'},
               ),
        row=1, col=1
    )

    # Add box plot trace to the second subplot
    fig.add_trace(_get_box_plot(df, col), row=2, col=1)

    # Update layout and axis labels
    fig.update_xaxes(title_text=gettext('Values'), row=2, col=1)
    fig.update_yaxes(title_text=gettext('Count'), row=1, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)
    fig.update_layout(bargap=0.1, showlegend=False)  # Histogram spacing

    fig.write_image('lixo.png')

    plotly_dict = fig.to_dict()
    del plotly_dict['layout']['template']
    analysis['result'] = plotly_dict


def _quantile_table(df: SparkDF, col: SparkCol, name: str, results: Dict,
                    analysis: Dict, total_count: int, i: int, is_numeric: bool,
                    version: tuple):
    qs = (.01, .05, .25, .5, .75, .95, .99)
    q = df.select(
        f.expr(f"percentile({name}, array{repr(qs)})")
    ).toPandas().to_numpy().tolist()[0][0]

    analysis['result'] = list(zip(qs, q))


def _frequency_table(df: SparkDF, col: SparkCol, name: str, results: Dict,
                     analysis: Dict, total_count: int, i: int, is_numeric: bool,
                     version: tuple):

    data = df.groupBy(col).count().orderBy(
        col).limit(20).toPandas().to_numpy().tolist()
    items, counts = list(zip(*data))

    fig = go.Figure(data=[go.Bar(x=items, y=counts)])
    fig.update_layout(
        title_text=gettext("Frequency Table"),
        xaxis_title_text=gettext("Items"),
        yaxis_title_text=gettext("Count")
    )

    plotly_dict = fig.to_dict()
    del plotly_dict['layout']['template']
    analysis['result'] = plotly_dict


def _summary_stats(df: SparkDF, columns: List[ColumnToSumarize],
                   total_count: int, version: tuple):

    types_actions = {
        "total_count": lambda column: f.count(column),
        "count": lambda column: f.count(column),
        "mean": lambda column: f.mean(column),
        "avg": lambda column: f.mean(column),
        "sum": lambda column: f.sum(column),
        "median": lambda column: (f.expr(
            f"percentile({column._jc.toString()}, 0.5)")),
        "std_dev": lambda column: f.stddev(column),
        "variance": lambda column: f.variance(column),
        "skewness": lambda column: f.skewness(column),
        "kurtosis": lambda column: f.kurtosis(column),
        "std_error_mean": lambda column: (f.stddev(column)
                                          / (total_count ** 0.5)),
        "n_finite": lambda column: f.count(column),
        "n_null": lambda column: total_count - f.count(column),
        "zero_count": lambda column: f.count(f.when(column == 0, True)),
        "zero_ratio": lambda column:
            (f.count(f.when(column == 0, True)) / total_count),
        "non_zero_ratio": lambda column:
            (1 - f.count(f.when(column == 0, True)) / total_count),
        "q1": lambda column: (f.expr(
            f"percentile({column._jc.toString()}, 0.25)")),
        "q3": lambda column: (f.expr(
            f"percentile({column._jc.toString()}, 0.75)")),
        "n_distinct": lambda column: f.approx_count_distinct(column),
        "min": lambda column: f.min(column),
        "max": lambda column: f.max(column),
        "range": lambda column: (f.max(column) - f.min(column)),
        "mode": lambda column: f.mode(column) if version >= (3, 4, 0)
            else f.lit(gettext('unsupported'))
    }
    any_type_stats = set([
        'count', 'total_count', 'min', 'max', 'n_distinct', 'n_null', 'mode'
    ])
    #import pdb; pdb.set_trace()

    stats = []
    results = []
    for i, col in enumerate(columns):
        name = col.spark_column._jc.toString()
        for fn in col.functions:
            if fn not in types_actions:
                print('Fn not found', fn)
            elif fn in any_type_stats or col.is_numeric:
                stats.append(
                    types_actions.get(fn)(col.spark_column).alias(
                        f'{fn}_{i}')
                )
                # Store where to save the final result
                results.append([fn, col.results])
            else:
                print(
                    f'Fn {fn} not supported for {name}')
    if stats:
        #import pdb;  pdb.set_trace()
        for value, [fn, result] in zip(
                list(df.select(*stats).first()), results):
            result[fn] = value

# column_names, types, params):


def generate_univariate_analysis(df, univariate, version):
    results = copy.deepcopy(univariate)

    total_count = None
    to_summarize = []
    # Count the size of the dataframe zero or once
    for i, attribute in enumerate(
            results.get('univariate', {}).get('attributes', [])):
        col = f.col(attribute.get('name'))
        # import pdb; pdb.set_trace()
        is_numeric = is_numeric_col(df.schema, attribute.get('name'))
        for analysis in attribute.get('analyses'):
            function_names = set([fn for fn in analysis.get('functions', [])])
            count_required = set(
                ['total_count', 'std_error_mean', 'zero_ratio',
                 'non_zero_ratio'])
            if (count_required.intersection(function_names)
                    and total_count is None):
                total_count = df.count()
            analisys_name = analysis.get('name')
            if analisys_name == 'summary_stats':
                # _summary_stats(df, col, attribute.get('name'),
                #                results, analysis, total_count,
                #                i, is_numeric, version)
                analysis['results'] = {}
                to_summarize.append(ColumnToSumarize(
                    col, attribute.get('name'), analysis.get('functions'),
                    analysis['results'], is_numeric))
            elif analisys_name == 'histogram':
                _histogram(df, col, attribute.get('name'),
                           results, analysis, total_count,
                           i, is_numeric, version)
            elif analisys_name == 'box_plot':
                _box_plot(df, col, analysis)
            elif analisys_name == 'frequency_table':
                _frequency_table(df, col, attribute.get('name'),
                                 results, analysis, total_count,
                                 i, is_numeric, version)
            elif analisys_name == 'quantile_table':
                _quantile_table(df, col, attribute.get('name'),
                                results, analysis, total_count,
                                i, is_numeric, version)

    if to_summarize:
        _summary_stats(df, to_summarize, total_count, version)

    return results


def analyse_data_source(payload):
    try:
        ds = get_data_source(payload, payload.get('id'))
        spark = (SparkSession.builder.master("local")
                                     .appName("analyse_data_source")
                                     .getOrCreate())

        schema = get_schema(ds)
        if ds_format == 'CSV':
            ds_format = ds.get('format')
            df = spark.read.csv(ds.get('url'),
                                header=ds.get('is_first_line_header'),
                                schema=schema)
        elif ds_format == 'PARQUET':
            df = spark.read.parquet(ds.get('url'))
        else:
            raise ValueError(
                gettext('Unsupported format: {}').format(ds_format))

        column_names = set(df.columns)

        analyses = ['univariate', 'bivariate', 'multivariate']

        # Force gettext tools to parse these terms
        analysis_text = [gettext('univariate'), gettext('bivariate'),
                         gettext('multivariate')]

        for i, analysis in enumerate(analyses):
            analysis_data = payload.get(analysis, {})
            types = analysis_data.get('types', {})
            if len(types) == 0:
                raise ValueError(
                    gettext('No analysis type specified for {}').format(
                        analysis_text[i]))
            analysis_columns = analysis_data.get('attributes') or []
            invalid = [c for c in analysis_columns if c not in column_names]

            if len(analysis_columns) == 0 or len(invalid) > 0:
                raise ValueError(gettext('Invalid attributes: {}').format(
                    ', '.join(invalid)))
    except Exception as e:
        log.error(e)
        return {'status': 'ERROR', 'message': gettext('Internal error'),
                'detail': str(e)}


def execute():
    spark = SparkSession.builder.appName("Test").getOrCreate()
    version = tuple([int(x) for x in spark.version.split('.')])

    df = spark.read.format("csv").option("header", "true").option(
        'inferSchema', 'true').load(
        "file:///scratch/walter/juicer/tests/data/iris.csv.gz")
    df.cache()
    payload = {
        'univariate': {
            'attributes': [
                {
                    'name': 'sepalwidth',
                    'analyses': [
                        {
                            'name': 'summary_stats',
                            'functions': [
                                'count',
                                'min',
                                'max',
                                'avg',
                            ]
                        },
                        {
                            'name': 'histogram'
                        },
                        {
                            'name': 'box_plot'
                        }
                    ]
                },
                {
                    'name': 'sepallength',
                    'analyses': [
                        {
                            'name': 'summary_stats',
                            'functions': [
                                'count',
                                'min',
                                'max',
                                'avg',
                                'q1',
                                'median',
                                'q3'
                            ]
                        }
                    ]
                },
                {
                    'name': 'class',
                    'analyses': [
                        {
                            'name': 'summary_stats',
                            'functions': [
                                'count',
                                'min',
                                'max',
                            ]
                        }
                    ]
                }
            ]
        }
    }
    print(generate_univariate_analysis(df, payload, version))


execute()
