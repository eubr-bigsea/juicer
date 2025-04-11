# coding=utf-8
import collections
import datetime
import decimal
import io
import itertools
import json
import math
import re
import functools
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd
import simplejson
from nltk import ngrams
from six import text_type
from gettext import gettext

# https://github.com/microsoft/pylance-release/issues/140#issuecomment-661487878
_: Callable[[str], str]


def is_numeric(schema, col):
    import pyspark.sql.types as spark_types
    from pyspark.ml.linalg import VectorUDT
    return isinstance(schema[str(col)].dataType, spark_types.NumericType) or \
        isinstance(schema[str(col)].dataType, VectorUDT)


def is_numeric_col(schema, col):
    import pyspark.sql.types as spark_types
    return isinstance(schema[str(col)].dataType, spark_types.NumericType)


def cast_value(schema, col, value):
    from datetime import datetime

    import pyspark.sql.types as spark_types
    field = schema[col]
    if isinstance(field.dataType, spark_types.StringType):
        return str(value)
    elif isinstance(field.dataType, spark_types.IntegralType):
        return int(value)
    elif isinstance(field.dataType, spark_types.FractionalType):
        return float(value)
    elif isinstance(field.dataType, spark_types.BooleanType):
        return bool(value)
    elif isinstance(field.dataType, spark_types.DateType):
        return datetime.strptime(value, '%Y-%m-%d')
    elif isinstance(field.dataType, spark_types.TimestampType):
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    else:
        raise ValueError(
            _('Unsupported value "{}" for attribute "{}"').format(
                value, col))


def default_encoder_sklearn(obj):
    if isinstance(obj, float):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        return str(obj)


def default_encoder(obj):
    from pyspark.ml.linalg import DenseVector
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, DenseVector):
        return list(obj)
    else:
        return str(obj)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if math.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        return super(NpEncoder, self).default(obj)


class SimpleJsonEncoder(simplejson.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return default_encoder(obj)


class SimpleJsonEncoderSklearn(simplejson.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return default_encoder_sklearn(obj)


class CustomEncoderSkLearn(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return default_encoder(list(obj))
        return default_encoder_sklearn(obj)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return default_encoder(list(obj))
        else:
            try:
                if np.isnan(obj):
                    return None
            except Exception:
                return f'{type(obj)} {str(obj)}'
        return default_encoder(obj)


def get_csv_schema(df, only_name=False):
    return ','.join(get_schema_fmt(df, only_name))


def get_csv_schema_sklearn(df, only_name=False):
    return ','.join(get_schema_fmt_sklearn(df, only_name))


def get_schema_fmt(df, only_name=False):
    if only_name:
        return [f.name for f in df.schema.fields]
    else:
        return ['{}:{}'.format(f.dataType, f.name) for f in df.schema.fields]


def get_schema_fmt_sklearn(df, only_name=False):
    if only_name:
        return list(df.columns)
    else:
        return ['{}:{}'.format(i, str(f))
                for i, f in zip(df.columns, df.dtypes)]


def get_dict_schema(df):
    return [dict(type=f.dataType, name=f.name) for f in df.schema.fields]


def with_column_index(sdf, name):
    import pyspark.sql.types as spark_types
    new_schema = spark_types.StructType(sdf.schema.fields + [
        spark_types.StructField(name, spark_types.LongType(), False), ])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(
        schema=new_schema)


def convert_to_csv(row):
    result = []
    for v in row:
        if isinstance(v, datetime.datetime):
            result.append(v.isoformat())
        elif isinstance(v, (str, text_type)):
            result.append('"{}"'.format(v))
        else:
            result.append(str(v))
    return ','.join(result)


def convert_to_python(row):
    result = []
    for v in row:
        t = type(v)
        if t in [datetime.datetime, datetime.date]:
            result.append(v.isoformat())
        else:
            result.append(v)
    return result


def format_row_for_visualization(row):
    date_types = [datetime.datetime, datetime.date]
    if len(row) == 2:
        # Use first column as id and name
        value = row[1] if type(
            row[1]) not in date_types else row[1].isoformat()
        _id = row[0]
        name = row[0]
    elif len(row) == 3:
        # Use first column as id and name
        value = row[2] if type(
            row[2]) not in date_types else row[2].isoformat()
        _id = row[0]
        name = row[0]
    else:
        raise ValueError(_('Invalid input data for visualization. '
                                 'It should contains 2 (name, value) or '
                                 '3 columns (id, name, value).'))
    return dict(id=_id, name=name, value=value)


def format_row_for_bar_chart_visualization(row):
    date_types = [datetime.datetime, datetime.date]
    if len(row) == 2:
        # Use first column as id and name
        value = row[1] if type(
            row[1]) not in date_types else row[1].isoformat()
        _id = row[0]
        name = row[0]
    elif len(row) == 3:
        # Use first column as id and name
        value = row[2] if type(
            row[2]) not in date_types else row[2].isoformat()
        _id = row[0]
        name = row[0]
    else:
        raise ValueError(_('Invalid input data for visualization. '
                                 'It should contains 2 (name, value) or '
                                 '3 columns (id, name, value).'))
    return dict(id=_id, name=name, value=value)


def emit_schema(task_id, df, emit_event, name, notebook=False):
    from juicer.spark.reports import SimpleTableReport
    headers = [_('Attribute'), _(
        'Type'), _('Metadata (Spark)')]
    rows = [[f.name, str(f.dataType), json.dumps(f.metadata) if f else ''] for f
            in df.schema.fields]
    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Schema for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Schema for {}').format(name),
               task={'id': task_id})


def emit_schema_sklearn_explorer(task_id, df, emit_event, name, notebook=False):
    rows = [{'name': c, 'type': d.name} for c, d in zip(df.columns, df.dtypes)]

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=json.dumps(rows),
               type='OBJECT', title=_('Schema for {}').format(name),
               meaning='schema',
               task={'id': task_id})


def emit_schema_sklearn(task_id, df, emit_event, name, notebook=False):
    from juicer.spark.reports import SimpleTableReport
    headers = [_('Attribute'), _('Type')]
    rows = [[i, str(f)] for i, f in zip(df.columns, df.dtypes)]
    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Schema for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Schema for {}').format(name),
               task={'id': task_id})


def emit_sample(task_id, df, emit_event, name, size=50, notebook=False,
                title=None):
    from juicer.spark.reports import SimpleTableReport
    headers = [f.name for f in df.schema.fields]

    number_types = (int, float, decimal.Decimal)

    rows = []
    for row in df.take(size):
        new_row = []
        rows.append(new_row)
        for col in row:
            if isinstance(col, (str, text_type)):
                value = col
            elif isinstance(col, (datetime.datetime, datetime.date)):
                value = col.isoformat()
            elif isinstance(col, number_types):
                value = str(col)
            else:
                value = json.dumps(col, cls=CustomEncoder)
            # truncate column if size is bigger than 200 chars.
            if len(value) > 200:
                value = value[:150] + ' ... ' + value[-50:]
            new_row.append(value)

    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    if title is None:
        title = _('Sample data for {}').format(name)
    content = SimpleTableReport(css_class, headers, rows, title, numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=title,
               task={'id': task_id})


def is_float_or_null(v):
    try:
        if v is None:
            return True
        float(v)
        return True
    except:
        return False


def emit_sample_data_explorer(task_id, df, emit_event, name, size=50,
                              notebook=False, describe=False, infer=False, use_types=None, page=1):
    pandas_df = df.toPandas()
    for c in df.schema.fields:
        # print(c.name, c.dataType.typeName())
        if c.dataType.typeName() in ('integer', 'byte', 'short', 'long'):
            pandas_df[c.name] = pandas_df[c.name].astype('Int64')
        elif c.dataType.typeName() in ('decimal'):
            pandas_df[c.name] = pandas_df[c.name].astype('float64')

    emit_sample_sklearn_explorer(task_id, pandas_df, emit_event, name,
                        size, notebook, describe, infer, use_types, page)


def emit_sample_explorer_polars(task_id, df, emit_event, name, size=50, notebook=False,
                        describe=False, infer=False, use_types=None, page=1):

    import polars as pl
    import polars.selectors as cs
    # Discard last '}' in order to include more information
    try:
        df_aux = df.with_columns([cs.by_dtype(pl.Date).dt.strftime('%Y-%m-%d')])
        result = [df_aux.limit(size).write_json(None)[:-1]]
    except Exception:
        raise ValueError(gettext('Internal error'))

    def get_generic_type(t):
        # Dict does not work because __hash__ implementation is not correct?
        if t == pl.List:
            return '"List"'
        elif t == pl.Datetime:
            return '"Datetime"'
        else:
            return f'"{str(t)}"'

    result.append(', "types": [' + (', '.join([
        get_generic_type(t)
        for t in df.dtypes])) + ']')
    result.append(
        f', "page": {page}, "size": {size}, "total": {df.shape[0]}, "format": "polars"')
    result.append(
        f', "estimated_size": {df.estimated_size()}')
    result.append('}')

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=' '.join(result), meaning='sample',
               type='OBJECT', title=_('Sample data for {}').format(name),
               task={'id': task_id})


def emit_sample_sklearn(task_id, df, emit_event, name, size=50, notebook=False,
                        describe=False, infer=False, use_types=None, page=1):
    from juicer.spark.reports import SimpleTableReport
    headers = list(df.columns)

    number_types = (int, float, decimal.Decimal)
    rows = []

    for row in df.head(size).values:
        new_row = []
        rows.append(new_row)
        for col in row:
            if isinstance(col, str):
                value = col
            elif isinstance(col, (datetime.datetime, datetime.date)):
                value = col.isoformat()
            elif isinstance(col, number_types):
                value = str(col)
            elif isinstance(col, list):
                value = '[' + ', '.join(
                    [str(x) if isinstance(x, number_types)
                     else "'{}'".format(x) for x in col]) + ']'
            else:
                value = json.dumps(col, cls=CustomEncoderSkLearn)
            # truncate column if size is bigger than 200 chars.
            if len(value) > 200:
                value = value[:150] + ' ... ' + value[-50:]
            new_row.append(value)

    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Sample data for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Sample data for {}').format(name),
               task={'id': task_id})


def emit_sample_sklearn_explorer(task_id, df, emit_event, name, size=50, notebook=False,
                        describe=False, infer=False, use_types=None, page=1):

    from collections import defaultdict

    import pandas as pd
    from pandas.api import types

    result = {}
    type_mappings = {'Int64': 'Integer', 'Float64': 'Decimal', 'object': 'Text',
                     'datetime64[ns]': 'Datetime', 'float64': 'Decimal',
                     'int64': 'Integer', 'bool': 'Boolean', 'array': 'Array'}

    df2 = df
    # Decide which data frame to use.
    # UI may require original data, represented as string and
    # without casting. But in order to describe the data set,
    # a inferred data frame (or a user provided types) must be
    # used.
    if describe and False:
        converters = {}
        if use_types:
            pandas_converters = {
                'Integer': lambda v: pd.to_numeric(v, errors='coerce'),
                'Decimal': lambda v: pd.to_numeric(v, errors='coerce'),
                'DateTime': lambda v: pd.to_datetime(v, errors='coerce'),
                'Boolean': bool,
            }
            # User provided types for attributes
            for attr, dtype in use_types.items():
                f = pandas_converters.get(dtype)
                if f:
                    converters[attr] = f
        if infer:
            # df.infer_objects() is not working.
            # save as CSV to infer data types
            buf = io.StringIO()
            work_df.to_csv(buf, index=False)
            buf.seek(0)
            df2 = pd.read_csv(buf, converters=converters)

    rows = []

    number_types = (int, float, decimal.Decimal)

    truncated = set()
    missing = defaultdict(list)
    invalid = defaultdict(list)

    dtypes = df2.dtypes[:]
    is_pandas_df = not hasattr(df, 'toPandas')
    if is_pandas_df:
        work_df = df
    else:
        work_df = df.limit(size * page).toPandas()

    it = work_df.iloc[size * (page - 1): size * page].iterrows()
    for y, (label, row) in enumerate(it):
        new_row = []
        for x, col in enumerate(work_df.columns):

            col_value = row[col]
            col_py_type = type(col_value)
            if (col_py_type != list and (
                    pd.isnull(row[col]) or (not row[col] and row[col] != 0))):
                missing[y].append(x)
                new_row.append('')
                continue

            if types.is_datetime64_dtype(work_df[col].dtypes):
                value = row[col].isoformat()
            elif types.is_numeric_dtype(col_py_type):
                if not types.is_integer_dtype(col_py_type):
                    value = round(row[col], 8)
                else:
                    value = row[col]
            elif types.is_datetime64_any_dtype(col_py_type):  # list of dates
                value = '[' + ','.join(['"{}'.format(d.isoformat())
                                        for d in row[col]]) + ']'
            elif (isinstance(col_value, Sequence) and
                  not isinstance(col_value, (str, bytes, bytearray))):
                value = '[' + ', '.join(
                    [str(x) if isinstance(x, number_types)
                     else "'{}'".format(x) for x in row[col]]) + ']'
                dtypes[x] = 'array'
            elif col_py_type == decimal.Decimal:
                value = str(row[col])
            elif col_py_type == datetime.date:
                value = row[col].strftime('%Y-%m-%d')
            elif types.is_string_dtype(col_py_type):
                # truncate column if size is bigger than 200 chars.
                value = row[col]
                if value and len(value) > 60:
                    value = value[:60] + ' (trunc.)'
                    truncated.add(col)
            else:
                value = json.dumps(row[col], cls=CustomEncoder)

            new_row.append(value)
        rows.append(new_row)

    result['attributes'] = [{'label': i, 'key': i,
                             'type': type_mappings.get(str(f), str(f))}
                            for i, f in zip(df2.columns, dtypes)]

    result['rows'] = rows
    result['page'] = page
    result['size'] = size
    result['total'] = len(work_df)
    if describe:
        missing_counters = df2.isna().sum().to_dict()
        result['total'] = len(work_df)
        for attr in result['attributes']:
            attr['missing_count'] = missing_counters.get(attr['key'], 0)
            attr['count'] = result['total']
            attr['invalid_count'] = 0

        pandas_converters = {
            'Integer': lambda v: v.isdigit(),
            'Decimal': lambda v: is_float_or_null(v),
            'DateTime': lambda v: True,
            'Boolean': lambda v: True,
        }
        for i, attr in enumerate(work_df.columns):
            f = None
            if pd.api.types.is_integer_dtype(work_df[attr].dtype):
                f = pandas_converters.get('Integer')
            elif pd.api.types.is_numeric_dtype(work_df[attr].dtype):
                f = pandas_converters.get('Decimal')

            if f:
                df_invalid = work_df[~work_df[attr].apply(
                    is_float_or_null) & work_df[attr].notnull()]
                result['attributes'][i]['invalid_count'] = len(df_invalid)
                invalid[attr].extend(df_invalid[:size].index.to_numpy())

        result['missing'] = missing
        result['invalid'] = invalid
        result['truncated'] = list(truncated)
    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=json.dumps(result), meaning='sample',
               type='OBJECT', title=_('Sample data for {}').format(name),
               task={'id': task_id})


def emit_sample_sklearn(task_id, df, emit_event, name, size=50, notebook=False,
                        describe=False, infer=False, use_types=None, page=1):
    from juicer.spark.reports import SimpleTableReport
    headers = list(df.columns)

    number_types = (int, float, decimal.Decimal)
    rows = []

    for row in df.head(size).values:
        new_row = []
        rows.append(new_row)
        for col in row:
            if isinstance(col, str):
                value = col
            elif isinstance(col, (datetime.datetime, datetime.date)):
                value = col.isoformat()
            elif isinstance(col, number_types):
                value = str(col)
            elif isinstance(col, list):
                value = '[' + ', '.join(
                    [str(x) if isinstance(x, number_types)
                     else "'{}'".format(x) for x in col]) + ']'
            else:
                value = json.dumps(col, cls=CustomEncoderSkLearn)
            # truncate column if size is bigger than 200 chars.
            if len(value) > 200:
                value = value[:150] + ' ... ' + value[-50:]
            new_row.append(value)

    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Sample data for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Sample data for {}').format(name),
               task={'id': task_id})


def analyse_attribute(task_id: str, df: Any, emit_event: Any, attribute: str,
                      msg: dict) -> None:

    stats = ['median', 'nunique']
    # import plotly.express as px
    import polars as pl
    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=True)
    # print('Analyse attribute', '*' * 20, isinstance(df, pl.DataFrame))
    if isinstance(df, pd.DataFrame):
        from pandas.api.types import is_numeric_dtype

        pandas_df = df.select(attribute).toPandas() if hasattr(df, 'toPandas') \
            else df
        analysis_type = 'table'
        if attribute is None:  # Statistics for the entire dataframe
            d = pandas_df.describe(include='all')
            d.append(pandas_df.reindex(d.columns, axis=1).agg(stats))
            result = d.transpose().to_json(orient="split", double_precision=4)
        elif msg.get('cluster'):
            from datasketch import MinHash, MinHashLSH
            lsh = MinHashLSH(
                threshold=msg.get('threshold', msg.get('similarity', 0.8)),
                num_perm=128)
            min_hashes = {}
            words = pandas_df[attribute].drop_duplicates().astype('str')
            print('*' * 30)
            print(words)
            print('*' * 30)
            for c, i in enumerate(words):
                min_hash = MinHash(num_perm=128)
                for d in ngrams(i, 3):
                    min_hash.update("".join(d).encode('utf-8'))
                lsh.insert(c, min_hash)
                min_hashes[c] = min_hash
            similar = []
            for k, i in enumerate(min_hashes.keys()):
                q = lsh.query(min_hashes[k])
                if len(q) > 1:
                    similar.append([words.iloc[i] for i in q])
            result = json.dumps(similar[:20])
            analysis_type = 'cluster'
        else:
            info = {}
            serie = pandas_df[attribute]
            stats = ['median', 'nunique', 'skew', 'var', 'kurtosis']
            if is_numeric_dtype(serie):
                d = serie.describe(include='all').append(serie.agg(stats))
            else:
                d = serie.describe(include='all')

            info['stats'] = dict(list(zip(d.index, d)))

            if is_numeric_dtype(serie):
                info['histogram'] = [x.tolist() for x in np.histogram(
                    serie.dropna(), bins=40)]

                q1 = info['stats']['25%']
                q3 = info['stats']['75%']
                iqr = q3 - q1
                info['stats']['iqr'] = iqr
                info['fence_low'] = q1 - 1.5 * iqr
                info['fence_high'] = q3 + 1.5 * iqr
                info['outliers'] = serie[
                    ((serie < info['fence_low'])
                     | (serie > info['fence_high']))].iloc[:10].tolist()

            counts = serie.value_counts(dropna=False).iloc[:20]
            info['top20'] = list(zip(
                [x if not pd.isna(x) else 'null' for x in counts.index],
                counts))
            info['nulls'] = serie.isna().sum()
            info['stats']['nulls'] = serie.isna().sum()
            info['stats']['rows'] = len(serie)

            # fig = px.histogram(serie,
            #           marginal="box", # or violin, rug
            #           )
            # info['plotly'] = fig.to_dict()

            # box_plot = io.BytesIO()
            # fig = serie.plot.box(figsize=(1,2))
            # plt.tight_layout()
            # fig.figure.savefig(box_plot, format='png')
            # box_plot.seek(0)
            # info['box_plot'] = base64.b64encode(box_plot.read()).decode('utf8')
            # plt.close()

            result = json.dumps(info, cls=NpEncoder)
            analysis_type = 'attribute'
    elif isinstance(df, pl.DataFrame):
        # Utility function to cast attributes and avoid type conflict
        def _cast(df: pl.DataFrame) -> pl.DataFrame:
            columns = []
            for s in df:
                if s.is_numeric() or s.is_boolean():
                    columns.append(s.cast(float).round(4))
                else:
                    columns.append(s)
            return pl.DataFrame(columns)

        polars_df = df.select(attribute) if attribute is not None \
            else df
        analysis_type = 'table'
        if attribute is None:  # Statistics for the entire dataframe
            metrics = [polars_df.mean, polars_df.var, polars_df.std,
                polars_df.min, polars_df.max,
                functools.partial(polars_df.quantile, .25),
                polars_df.median,
                functools.partial(polars_df.quantile, .75),
            ]
            result = pl.concat([_cast(m()) for m in metrics])
            names = [gettext('mean'), gettext('var'), gettext('std'),
                     gettext('min'), gettext('max'), gettext('25%'),
                     gettext('median'), gettext('75%'), ]

            result = result.select(
                [pl.col(c).cast(pl.Utf8) for c in result.columns]
                ).transpose(include_header=False, column_names=names)
            df_uniq = (
                polars_df.select(
                     [pl.when(dtype == pl.List).then(-1).otherwise(
                        pl.col(col).n_unique()).alias(col)
                        for (col, dtype) in
                        zip(polars_df.columns, polars_df.dtypes)])
                        .transpose(column_names=['unique']))

            result = pl.concat([result, df_uniq], how='horizontal')

            result.insert_at_idx(
                0, pl.Series(gettext('attribute'), polars_df.columns)
            )
            cast_types = [pl.Utf8, pl.Float64, pl.Float64, pl.Float64,
                pl.Utf8, pl.Utf8, pl.Float64, pl.Float64, pl.Float64, pl.Int64]
            result = result.with_columns([
                pl.col(c).cast(cast_types[i]).round(4).alias(c)
                if cast_types[i] == pl.Float64
                else pl.col(c)
                    for i, c in enumerate(result.columns)
            ])
            obj_result = json.loads(result.write_json(None, row_oriented=False))
            # Computes correlation
            import polars.selectors as cs
            attr_names = df.select(cs.numeric()).columns
            pairs = list(itertools.product(attr_names, attr_names))
            correlation = [x if not np.isnan(x) else None
                for x in [round(x, 4) for x in df.select([
                pl.corr(*v, method="spearman").alias(str(v)) for v in pairs])
                .row(0)
            ] ]
            attr_count = len(attr_names)
            final_corr = [correlation[i:i + attr_count] for i in
                     range(0, attr_count**2, attr_count)]
            numeric = [s.is_numeric() or s.is_boolean() for s in df]

            result = json.dumps({
                'table': obj_result,
                'correlation': final_corr,
                'attributes': list(zip(df.columns, numeric)),
                'numeric': [s.name for s in df if s.is_numeric()]
            }, cls=CustomEncoder)


        elif msg.get('cluster'):
            from datasketch import MinHash, MinHashLSH
            lsh=MinHashLSH(
                threshold = msg.get('threshold', msg.get('similarity', 0.8)),
                num_perm = 128)
            min_hashes={}
            words=pandas_df[attribute].drop_duplicates().astype('str')
            print('*' * 30)
            print(words)
            print('*' * 30)
            for c, i in enumerate(words):
                min_hash=MinHash(num_perm = 128)
                for d in ngrams(i, 3):
                    min_hash.update("".join(d).encode('utf-8'))
                lsh.insert(c, min_hash)
                min_hashes[c]=min_hash
            similar=[]
            for k, i in enumerate(min_hashes.keys()):
                q=lsh.query(min_hashes[k])
                if len(q) > 1:
                    similar.append([words.iloc[i] for i in q])
            result=json.dumps(similar[:20])
            analysis_type='cluster'
        else:  # statistics for a single attribute
            series=df.get_column(attribute)
            if series.is_numeric() and series.dtype != pl.Boolean:
                df=df.with_columns([pl.col(attribute).cast(pl.Float64)])
                series=df.get_column(attribute)
                names=[gettext(n) for n in
                         ['mean', 'var', 'std', 'min', 'max',  '25%', 'median',
                          '75%', 'unique', 'skew', 'kurtosis', 'count', 'nulls']]
                metrics= [polars_df.mean, polars_df.var, polars_df.std,
                    polars_df.min, polars_df.max,
                    functools.partial(polars_df.quantile, .25),
                    polars_df.median,
                    functools.partial(polars_df.quantile, .75),
                ]
                result = pl.concat([_cast(m()) for m in metrics])
                extra = polars_df.select([
                        pl.n_unique(attribute).alias('unique'),
                        pl.col(attribute).skew().round(4).alias('skew'),
                        pl.col(attribute).kurtosis().round(4).alias('kurtosis'),
                        pl.col(attribute).count().alias('count'),
                        pl.col(attribute).null_count().alias('nulls'),
                        ]).transpose(column_names=[attribute])
                result = pl.concat([result, extra], how='vertical')
                info = {'stats': {n:v for n, v in
                                  zip(names, result.get_column(attribute))}}
                info['histogram'] = [x.tolist() for x in np.histogram(
                     series.drop_nulls(), bins=40)]
                q1 = info['stats']['25%']
                q3 = info['stats']['75%']
                iqr = q3 - q1
                info['stats']['iqr'] = iqr
                info['fence_low'] = q1 - 1.5 * iqr
                info['fence_high'] = q3 + 1.5 * iqr
                info['outliers'] = (df.select(attribute)
                    .filter((pl.col(attribute) < info['fence_low'])
                     | (pl.col(attribute) > info['fence_high']))
                     .unique()
                     .limit(10)
                     .select(pl.col(attribute).round(4))
                     .get_column(attribute)
                     .to_numpy().tolist())
                info['top20'] = (polars_df
                    .groupby(attribute)
                    .agg(pl.col(attribute).count().alias('counts'))
                    .sort('counts', descending=True)
                    .limit(20)
                    .select([pl.col(attribute).fill_null('null'), 'counts'])
                    .to_numpy()
                    .tolist()
                )

            else: # Non-numeric attributes
                names = [gettext(n) for n in
                         ['count', 'nulls', 'min', 'max', 'mode', 'unique']]
                metrics = [
                    polars_df.min, polars_df.max
                ]
                extra = polars_df.select([
                        pl.col(attribute).count().alias('count'),
                        pl.col(attribute).null_count().alias('nulls'),
                        pl.col(attribute).drop_nulls().cast(
                            pl.Utf8).min().alias('min'),
                        pl.col(attribute).drop_nulls().cast(
                            pl.Utf8).max().alias('max'),
                        pl.col(attribute).drop_nulls().cast(
                            pl.Utf8).mode().alias('mode'),
                        pl.col(attribute).n_unique().alias('unique'),
                        ])
                result = extra
                info = {'stats': {n:v for n, v in
                                  zip(names, extra.to_numpy().tolist()[0] )}}

                info['histogram'] = list(zip(*df.select(attribute)
                    .drop_nulls()
                    .groupby(attribute, maintain_order=True)
                    .count().sort('count', descending=True)
                    .limit(40)
                    .rows()))[::-1]
            info['numeric'] = series.is_numeric()
            # if is_numeric_dtype(serie):
            # counts = serie.value_counts(dropna=False).iloc[:20]
            # info['top20'] = list(zip(
            #     [x if not pd.isna(x) else 'null' for x in counts.index],
            #     counts))

            # fig = px.histogram(serie,
            #           marginal="box", # or violin, rug
            #           )
            # info['plotly'] = fig.to_dict()

            # box_plot = io.BytesIO()
            # fig = serie.plot.box(figsize=(1,2))
            # plt.tight_layout()
            # fig.figure.savefig(box_plot, format='png')
            # box_plot.seek(0)
            # info['box_plot'] = base64.b64encode(box_plot.read()).decode('utf8')
            # plt.close()

            result = json.dumps(info, cls=NpEncoder)
            analysis_type = 'attribute'

    emit_event('analysis', status='COMPLETED',
               identifier=task_id,
               message=result,
               analysis_type=analysis_type,
               attribute=attribute,
               type='OBJECT',
               title=gettext('Analysis for attribute {}').format(attribute),
               task={'id': task_id})


class LazySparkTransformationDataframe(object):
    """
     Wraps a Spark Model in order to perform lazy transformation
    """

    def __init__(self, model, df, load_op):
        self.model = model
        self.df = df
        self.transformed_df = None
        self.load_op = load_op

    def __getitem__(self, item):
        if self.transformed_df is None:
            self.transformed_df = self.load_op(self.df)
        return self.transformed_df[item]

    def __getattr__(self, name):
        if self.transformed_df is None:
            self.transformed_df = self.load_op(self.df)
        return getattr(self.transformed_df, name)


class SparkObjectProxy(object):
    """
     Used to wrap Spark objects's methods calls and execute custom code before
     and/or after original method execution
    """

    def __init__(self, data_frame, **kwargs):
        """
        Wraps PySpark objects's methods with an implementation.
          :param data_frame Object being wrapped
          :param kwargs Contains method's name and a tuple with new
                        implementation (before/after, if None, ignored).
        """
        self.wrapped_obj = data_frame
        self.methods = kwargs

    def __getattr__(self, name):
        member_to_call = getattr(self.wrapped_obj, name)

        def wrapper(*args, **kwargs):
            if name in self.methods or '__any__' in self.methods:
                target = self.wrapped_obj

                # before method call
                if all(['__any__' in self.methods,
                        self.methods['__any__'][0] is not None]):
                    r = self.methods['__any__'][0](self.wrapped_obj)
                    if r is not None:  # side-effect, new target
                        target = r

                if name in self.methods and self.methods[name][0] is not None:
                    r = self.methods[name][0](self.wrapped_obj)
                    if r is not None:  # side-effect, new target
                        target = r

                method_to_call = getattr(target, name)
                if isinstance(method_to_call, collections.Callable):
                    result = method_to_call(*args, **kwargs)
                else:
                    result = method_to_call

                # after method call
                if all(['__any__' in self.methods,
                        self.methods['__any__'][1] is not None]):
                    r = self.methods['__any__'][1](self.wrapped_obj)
                    if r is not None:
                        result = r

                if name in self.methods and self.methods[name][1] is not None:
                    r = self.methods[name][1](self.wrapped_obj)
                    if r is not None:
                        result = r
                return result
            else:
                method_to_call = getattr(self.wrapped_obj, name)
                if isinstance(method_to_call, collections.Callable):
                    return method_to_call(*args, **kwargs)
                else:
                    return method_to_call

        return (wrapper if isinstance(member_to_call, collections.Callable)
                else member_to_call)


def spark_version(spark_session):
    return tuple(map(int, spark_session.version.split('.')))


def merge_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def handle_spark_exception(e):
    from pyspark.sql.utils import AnalysisException, IllegalArgumentException
    result = False

    if isinstance(e, AnalysisException):
        value_expr = re.compile(r'[`"](.+)[`"].+columns:\s(.+)$')
        found = value_expr.findall(e.desc.split('\n')[0])
        if found:
            field, fields = found[0]
            raise ValueError(
                _('Attribute {} not found. Valid attributes: {}').format(
                    field, fields.replace(';', '')))
        else:
            err_desc = e.desc.split('\n')[0]
            value_expr = re.compile(r'The data type of the expression in the '
                                    r'ORDER BY clause should be a numeric type')
            found = value_expr.findall(err_desc)
            if found:
                raise ValueError(
                    _('When using Window Operation with range type, '
                            'the order by attribute must be numeric.'))
            found = 'This Range Window Frame only accepts ' \
                    'at most one ORDER BY' in e.desc
            if found:
                raise ValueError(
                    _('When using Window Operation with range type, the '
                            'order option must include only one attribute.'))
            found = 'Path does not exist' in e.desc
            if found:
                raise ValueError(
                    _('Data source does not exist. It may have been '
                            'deleted.'))
            value_expr = re.compile(r'Table or view not found: (.+?);')
            found = value_expr.findall(err_desc)
            if found:
                raise ValueError(
                    _('Table or view not found: {}').format(found[0]))

            value_expr = re.compile(r'The table or view `(.+?)` cannot be found')
            found = value_expr.findall(err_desc)
            if found:
                raise ValueError(
                    _('Table or view not found: {}').format(found[0]))
            found = re.findall(
                r'Cannot resolve column name "(.+)" among (.+)',
                err_desc)
            if found:
                raise ValueError(
                    f'{_("Attribute")} {found[0][0]} '
                    f'{_("not found. Valid ones:")} {found[0][1]}.')
            found = re.findall(r'Invalid view name: (.+);', err_desc)
            if found:
                raise ValueError(f'Nome inválido para view: {found[0]}.')
    elif isinstance(e, KeyError):
        value_expr = re.compile(r'No StructField named (.+)\'$')
        found = value_expr.findall(str(e))
        if found:
            raise ValueError(
                _('Attribute {} not found.').format(found[0]))
    elif isinstance(e, IllegalArgumentException):
        # Invalid column type
        if 'must be of type equal' in str(e):
            value_expr = re.compile(
                "requirement failed: Column (.+?) must be"
                ".+following types: \[(.+?)\] but was actually of type (.+).")
            found = value_expr.findall(e.desc)
            if found:
                attr, correct, used = found[0]
                raise ValueError(
                    _('Attribute {attr} must be one of these types'
                            ' [{correct}], but it is {used}').format(
                        attr=attr, used=used, correct=correct
                    ))
        elif 'Available fields' in str(e):
            value_expr = re.compile(
                r'Field "(.+?)" does not exist.\nAvailable fields: (.+)',
                re.MULTILINE)
            found = value_expr.findall(e.desc)
            if found:
                used, correct = found[0]
                raise ValueError(
                    _(
                        'Attribute {} not found. Valid attributes: {}').format(
                        used, correct))
        elif 'requirement failed: A & B Dimension mismatch' in str(e):
            raise ValueError(
                gettext(
                    'Number of features and neurons in input layer do not match')
            )
        elif 'Binomial family only supports' in str(e):
            value_expr = re.compile(
                r'outcome classes but found (\d+)',
                re.MULTILINE)
            found = value_expr.findall(e.desc)
            if found:
                total = found[0]
                raise ValueError(
                    _('Binomial family only supports 1 or 2 outcome '
                            'classes but found {}').format(total))
        else:
            raise ValueError(e.desc) from None
    elif hasattr(e, 'java_exception'):
        se = 'org.apache.spark.SparkException'
        cause = e.java_exception.getCause()
        if cause and cause.getClass().getName() != se:
            if 'unwrapRemoteException' in dir(cause):
                cause = cause.unwrapRemoteException()
            else:
                while cause is not None and cause.getCause() is not None:
                    cause = cause.getCause()
        if cause is not None:
            nfe = 'java.lang.NumberFormatException'
            uoe = 'java.lang.UnsupportedOperationException'
            npe = 'java.lang.NullPointerException'
            bme = 'org.apache.hadoop.hdfs.BlockMissingException'
            ace = 'org.apache.hadoop.security.AccessControlException'
            iae = 'java.lang.IllegalArgumentException'
            rte = 'java.lang.RuntimeException'

            cause_msg = cause.getMessage() or ''
            inner_cause = cause.getCause()
            if cause.getClass().getName() == nfe and cause_msg:
                value_expr = re.compile(r'.+"(.+)"')
                value = value_expr.findall(cause_msg)[0]
                raise ValueError(_('Invalid numeric data in at least one '
                                         'data source (value: {})').format(
                    value).encode('utf8'))
            elif 'Malformed' in cause_msg:
                raise ValueError(_('At least one input data source is not '
                                         'in the correct format.'))
            elif inner_cause and inner_cause.getClass().getName() == npe:
                if cause_msg and 'createTransformFunc' in cause_msg:
                    raise ValueError(
                        _('There is null values in your data '
                                'set and Spark cannot handle them. '
                                'Please, remove them before applying '
                                'a data transformation.'))
                pass
            elif cause.getClass().getName() == rte:
                rte_msg = cause_msg
                if "Labels MUST be in {0, 1}" in cause_msg:
                    rte_msg = gettext(
                        "Algorithm (or output layer) supports only 2 classes, "
                        "but the input data have more that 2."
                    )
                elif "Vector values MUST be in {0, 1}" in cause_msg:
                    rte_msg = gettext(
                        "The algorithm or parameter is applicable only to "
                        "binary classification, but the input data have more "
                        "that 2 classes."
                    )
                raise ValueError(rte_msg)
            elif cause.getClass().getName() == bme:
                raise ValueError(
                    _(
                        'Cannot read data from the data source. In this case, '
                        'it may be a configuration problem with HDFS. '
                        'Please, check if HDFS namenode is up and you '
                        'correctly configured the option '
                        'dfs.client.use.datanode.hostname in Juicer\' config.'))
            elif cause.getClass().getName() == ace:
                raise ValueError(
                    _(
                        'You do not have permissions to read or write in the '
                        'storage. Probably, it is a configuration problem. '
                        'Please, contact the support.')
                )
            elif cause.getClass().getName() == iae:
                gbt_error = 'dataset with invalid label'
                if cause_msg is not None and gbt_error in cause_msg:
                    raise ValueError(_('GBT classifier requires labels '
                                             'to be in [0, 1] range.'))
                else:
                    raise ValueError(cause_msg)
        elif e.java_exception.getMessage():
            cause_msg = e.java_exception.getMessage()
            if 'already exists' in cause_msg:
                raise ValueError(
                    _('File already exists. Try to use options '
                            'to overwrite it.'))
            value_expr = re.compile(r'CSV data source does not support '
                                    r'(.+?) data type')
            value = value_expr.findall(cause_msg)
            if value:
                raise ValueError(
                    _(
                        'CSV format does not support the data type {}. '
                        'Try to convert the attribute to string '
                        '(see to_json()) before saving.'.format(value[0])))
    return result


def df_zip_with_index(df, offset=1, name="row_id"):
    import pyspark.sql.types as spark_types
    new_schema = spark_types.StructType(
        [spark_types.StructField(name, spark_types.LongType(),
                                 True)] + df.schema.fields
    )

    zipped_rdd = df.rdd.zipWithIndex()

    return zipped_rdd.map(
        lambda row_row_id: ([row_row_id[1] + offset] +
                            list(row_row_id[0]))).toDF(new_schema)
