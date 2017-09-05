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
        else:
            return str(obj)


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


def emit_schema(task_id, df, emit_event, name):
    from juicer.spark.reports import SimpleTableReport
    headers = ['Attribute', 'Type']
    rows = [[f.name, str(f.dataType)] for f in df.schema.fields]
    content = SimpleTableReport(
        'table table-striped table-bordered', headers, rows,
        'Schema for {}'.format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title='Schema for {}'.format(name),
               task=task_id)


def emit_sample(task_id, df, emit_event, name, size=50):
    from juicer.spark.reports import SimpleTableReport
    headers = [f.name for f in df.schema.fields]
    rows = [[str(col) for col in row] for row in df.take(size)]

    content = SimpleTableReport(
        'table table-striped table-bordered', headers, rows,
        'Sample data for {}'.format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title='Sample data for {}'.format(name),
               task=task_id)


class LazySparkTransformationDataframe(object):
    """
     Wraps a Spark Model in order to perform lazy transformation
    """

    def __init__(self, model, df, load_op):
        self.model = model
        self.df = df
        self.transformed_df = None
        self.load_op = load_op

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
                if callable(method_to_call):
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
                if callable(method_to_call):
                    return method_to_call(*args, **kwargs)
                else:
                    return method_to_call

        return wrapper if callable(member_to_call) else member_to_call
