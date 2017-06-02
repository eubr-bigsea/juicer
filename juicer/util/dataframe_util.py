# coding=utf-8
import decimal
import json

import datetime


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()


def get_csv_schema(df, only_name=False):
    return ','.join(get_schema_fmt(df, only_name))


def get_schema_fmt(df, only_name=False):
    if only_name:
        return [f.name for f in df.schema.fields]
    else:
        return ['{}:{}'.format(f.dataType, f.name) for f in df.schema.fields]


def get_dict_schema(df):
    return [dict(type=f.dataType, name=f.name) for f in df.schema.fields]


def convert_to_csv(row):
    result = []
    for v in row:
        t = type(v)
        if t in [datetime.datetime]:
            result.append(v.isoformat())
        elif t in [unicode, str]:
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
        value = row[1] if type(row[1]) not in date_types else row[1].isoformat()
        _id = row[0]
        name = row[0]
    elif len(row) == 3:
        # Use first column as id and name
        value = row[2] if type(row[2]) not in date_types else row[2].isoformat()
        _id = row[0]
        name = row[0]
    else:
        raise ValueError('Invalid input data for visualization. '
                         'It should contains 2 (name, value) or '
                         '3 columns (id, name, value).')
    return dict(id=_id, name=name, value=value)


def format_row_for_bar_chart_visualization(row):
    date_types = [datetime.datetime, datetime.date]
    if len(row) == 2:
        # Use first column as id and name
        value = row[1] if type(row[1]) not in date_types else row[1].isoformat()
        _id = row[0]
        name = row[0]
    elif len(row) == 3:
        # Use first column as id and name
        value = row[2] if type(row[2]) not in date_types else row[2].isoformat()
        _id = row[0]
        name = row[0]
    else:
        raise ValueError('Invalid input data for visualization. '
                         'It should contains 2 (name, value) or '
                         '3 columns (id, name, value).')
    return dict(id=_id, name=name, value=value)
