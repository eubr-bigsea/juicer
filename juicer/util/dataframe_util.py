# coding=utf-8
import datetime


def get_csv_schema(df):
    return ','.join(get_schema_fmt(df))


def get_schema_fmt(df):
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
