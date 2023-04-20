# coding=utf-8
import json
import os
import uuid
from textwrap import dedent, indent
from gettext import gettext

from juicer.service import limonero_service

from urllib.parse import urlparse, parse_qs
import juicer.scikit_learn.data_operation as sk


class DataReaderOperation(sk.DataReaderOperation):

    LIMONERO_TO_POLARS_DATA_TYPES = {
        "CHARACTER": 'pl.Utf8',
        "DATETIME": 'pl.Datetime',
        "DATE": 'pl.Date',
        "DOUBLE": 'pl.Float64',
        "DECIMAL": 'pl.Float64',
        "FLOAT": 'pl.Float32',
        "LONG": 'pl.Int64',
        "INTEGER": 'pl.Int32',
        "TEXT": 'pl.Utf8',
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        sk.DataReaderOperation.__init__(self, parameters, named_inputs,
                                        named_outputs)
        self.transpiler_utils.add_import('import gzip')

    def generate_code(self):
        """
        """
        if not self.has_code:
            return ''

        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        do_not_infer = self.infer_schema == self.DO_NOT_INFER
        mode_failfast = self.mode == self.OPT_MODE_FAILFAST

        protect = (self.parameters.get('export_notebook', False) or
                   self.parameters.get('plain', False)) or self.plain
        data_format = self.metadata.get('format')

        parsed = urlparse(self.metadata['url'])

        extra_params = {}
        if 'extra_params' in self.metadata['storage']:
            if self.metadata['storage']['extra_params']:
                extra_params = json.loads(self.metadata['storage'][
                    'extra_params'])

        if parsed.scheme not in ('hdfs', 'file', 'mysql'):
            raise ValueError(gettext('Scheme {} not supported').format(
                parsed.scheme))

        if data_format not in ('CSV', 'TEXT', 'PARQUET', 'JDBC', 'JSON'):
            raise ValueError(gettext('Format {} not supported').format(
                data_format))

        if data_format == 'JDBC':
            qs_parsed = parse_qs(parsed.query)
            if parsed.scheme not in self.SUPPORTED_DRIVERS:
                raise ValueError(
                    gettext('Database {} not supported').format(parsed.scheme))
            if not self.metadata.get('command'):
                raise ValueError(
                    gettext('No command nor table specified for data source.'))

            jdbc_code = indent(
                dedent(self.SUPPORTED_DRIVERS[parsed.scheme].format(
                    scheme=parsed.scheme, host=parsed.hostname,
                    db=parsed.path[1:],
                    port=parsed.port,
                    query=self.metadata.get('command'),
                    user=qs_parsed.get('user', [''])[0],
                    password=qs_parsed.get('password', [''])[0],
                    out=self.output)), '    ')
        else:
            jdbc_code = None

        self.header = self.metadata.get('is_first_line_header')

        attributes, converters, parse_dates, names = self.analyse_attributes(
            self.LIMONERO_TO_POLARS_DATA_TYPES,
            self.metadata.get('attributes'))

        self.template = """
        {%- if infer_from_limonero %}
        {%-   if attributes and format in ('TEXT', 'CSV') %}
        columns = {
        {%-     for attr in attributes %}
            '{{attr[0]}}': {{attr[1]}},
        {%-     endfor %}
        }
        {%-   elif format in ('TEXT', 'CSV') %}
        columns = {'value': object}
        {%-   endif %}
        {%- elif infer_from_data and format in ('TEXT', 'CSV') %}
        columns = None
        header = 'infer'
        {%- elif do_not_infer and format in ('TEXT', 'CSV') %}
        header = 'infer'
        {%- endif %}

        # Open data source
        {%- if protect %}
        f = open('{{parsed.path.split('/')[-1]}}', 'rb')
        {%- elif parsed.scheme == 'hdfs'  %}
        fs = pa.hdfs.connect(host='{{parsed.hostname}}', 
            port={{parsed.port}},
            user='{{extra_params.get('user', parsed.username) or 'hadoop'}}')
        f = fs.open('{{parsed.path}}', 'rb')
        {%- elif parsed.scheme == 'file' %}
        f = '{{parsed.path}}'
        {%- endif %}

        {%- if format == 'CSV' %}
        {{output}} = pl.read_csv(
            f, separator='{{sep}}',
            encoding='{{encoding}}',
            has_header={{header}},
            {%- if infer_from_limonero %}
            {%- if header %}
            new_columns={{names}},
            {%- endif %}
            dtypes=columns,
            try_parse_dates={{parse_dates}},
            {%- elif do_not_infer %}
            try_parse_dates = None,
            converters = None,
            dtypes=['str'],
            {%-   endif %}
            null_values={{na_values}},
            ignore_errors=True
            ).lazy()
        {%- if parsed.scheme == 'hdfs'  %}
        f.close()
        {%- endif %}
        {%-   if header == 'infer' %}
        {{output}}.columns = ['attr{{i}}'.format(i=i) 
                        for i, _ in enumerate({output}.columns)]
        {%-   endif %}
        {%- elif format == 'TEXT' %}
        {{output}} = pl.read_csv(f, separator='{{sep}}',
                                 encoding='{{encoding}}',
                                 compression='infer',
                                 names = ['value'],
                                 error_bad_lines={{mode_failfast}})
        f.close()
        {%- elif format == 'PARQUET' %}
        {{output}} = pl.read_parquet(f, engine='pyarrow')
        f.close()
        {%- elif format == 'JSON' %}
        {{output}} = pl.read_json(f, orient='records')
        f.close()
        {%- elif format == 'JDBC' %}
        {{jdbc_code}}
        {%- endif %}

        {%- if infer_from_data %}
        {{output}} = {{output}}.infer_objects()
        {%- endif %}
        
        """
        ctx = {
            'attributes': attributes,
            'parse_dates': True,
            'names': names,
            'converters': converters,

            'infer_from_limonero': infer_from_limonero,
            'infer_from_data': infer_from_data,
            'do_not_infer': do_not_infer,
            'is_first_line_header': self.header,

            'protect': protect,  # Hide information about path
            'parsed': parsed,
            'extra_params': extra_params,
            'format': data_format,
            'encoding': ((self.metadata.get('encoding', 'utf8') or 'utf8')
                         .replace('-', '')),
            'header': bool(self.header),
            'sep': self.sep,
            'na_values': self.null_values if len(self.null_values) else 'None',
            'output': self.output,
            'mode_failfast': mode_failfast,

            'jdbc_code': jdbc_code,
        }
        return dedent(self.render_template(ctx))


class SaveOperation(sk.SaveOperation):
    """
    Saves the content of the DataFrame at the specified path.
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_ID_PARAM = 'storage'
    FORMAT_PARAM = 'format'
    TAGS_PARAM = 'tags'
    OVERWRITE_MODE_PARAM = 'mode'
    HEADER_PARAM = 'header'

    MODE_ERROR = 'error'
    MODE_APPEND = 'append'
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'

    FORMAT_PICKLE = 'PICKLE'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'
    FORMAT_PARQUET = 'PARQUET'

    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    PANDAS_TO_LIMONERO_DATA_TYPES = {
        'object': "CHARACTER",
        'datetime64[ns]': "DATETIME",
        'float64': "DOUBLE",
        'float32': "FLOAT",
        'int64': "LONG",
        'Int64': "INTEGER",
        'int32': "INTEGER",
        'int16': "INTEGER",
        'int8': "INTEGER",
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        sk.SaveOperation.__init__(
            self, parameters, named_inputs, named_outputs)

    def generate_code(self):

        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']
        url = limonero_config['url']
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(
            url, token, self.storage_id)

        if storage['type'] != 'HDFS':
            raise ValueError(gettext('Storage type not supported: {}').format(
                storage['type']))

        protect = (self.parameters.get('export_notebook', False) or
                   self.parameters.get('plain', False)) or self.plain

        if storage['url'].endswith('/'):
            storage['url'] = storage['url'][:-1]
        if self.path.endswith('/'):
            self.path = self.path[:-1]

        if self.path.startswith('/'):
            self.path = self.path[1:]

        final_url = '{}/limonero/user_data/{}/{}/{}'.format(
            storage['url'],
            self.user['id'],
            self.path,
            self.name.replace(' ', '_'))

        if self.format == self.FORMAT_CSV and not final_url.endswith('.csv'):
            final_url += '.csv'
        elif self.format == self.FORMAT_JSON and not final_url.endswith(
                '.json'):
            final_url += '.json'
        elif self.format == self.FORMAT_PARQUET and not final_url.endswith(
                '.parquet'):
            final_url += '.parquet'

        parsed = urlparse(final_url)

        df_input = self.named_inputs['input data']
        extra_params = storage.get('extra_params') \
            if storage.get('extra_params') is not None else "{}"
        extra_params = json.loads(extra_params)

        hdfs_user = extra_params.get('user', parsed.username) or 'hadoop'
        self.template = """
            path = '{{path}}'
            {%- if scheme == 'hdfs' and not protect %}
            fs = pa.hdfs.connect(host='{{hdfs_server}}', 
                                 port={{hdfs_port}},
                                 user='{{hdfs_user}}')
            exists = fs.exists(path)
            {%- elif scheme == 'file' or protect %}
            exists = os.path.exists(path)
            {%- endif %}

            mode = '{{mode}}'
            if mode not in ('error', 'ignore', 'overwrite'):
                raise ValueError('{{error_invalid_mode}}')
            if exists:
                if mode == 'error':
                    raise ValueError('{{error_file_exists}}')
                elif mode == 'ignore':
                    emit_event(name='update task',
                        message='{{warn_ignored}}',
                        status='COMPLETED',
                        identifier='{{task_id}}')
                else:
                    {%- if scheme == 'hdfs' and not protect %}
                        fs.delete(path, False)
                    {%- elif scheme == 'file' or protect %}
                        os.remove(path)
                        parent_dir = os.path.dirname(path)
                        if not os.path.exists(parent_dir):
                            os.makedirs(parent_dir)
                    {%- endif %}
            else:
                {%-if scheme == 'hdfs' and not protect %}    
                fs.mkdir(os.path.dirname(path))
                {%- elif scheme == 'file' %}
                parent_dir = os.path.dirname(path)
                os.makedirs(parent_dir)
                {%- else %}
                pass
                {%- endif%}
            
            {%- if format == FORMAT_CSV %}
            {%- if scheme == 'hdfs' and not protect %}
            from io import StringIO
            with fs.open(path, 'wb') as f:
                s = StringIO()
                {{input}}.to_csv(s, separator=str(','), mode='w',
                header={{header}}, index=False, encoding='utf-8')
                f.write(s.getvalue().encode())               
            {%- elif scheme == 'file' or protect %}
            {{input}}.to_csv(path, separator=str(','), mode='w',
            header={{header}}, index=False, encoding='utf-8')
            {%- endif %}
            
            {%- elif format == FORMAT_PARQUET %}
            {%- if scheme == 'hdfs' and not protect %}
            from io import BytesIO
            with fs.open(path, 'wb') as f:
                s = BytesIO()
                {{input}}.to_parquet(s, engine='pyarrow')
                f.write(s.getvalue())               
            {%- elif scheme == 'file' or protect %}
            {{input}}.to_parquet(path, engine='pyarrow')
            {%- endif %}
            
            {%- elif format == FORMAT_JSON %}
            {%- if scheme == 'hdfs' and not protect %}
            from io import StringIO
            with fs.open(path, 'wb') as f:
                s = StringIO()
                {{input}}.to_json(s, orient='records')
                f.write(s.getvalue().encode())             
            {%- elif scheme == 'file' or protect %}
            {{input}}.to_json(path, orient='records')
            {%- endif %}
            {%- endif %}
            
            {%-if not protect %}
            # Code to update Limonero metadata information
            from juicer.service.limonero_service import register_datasource
            types_names = {{data_types}}

            write_header = {{header}}
            attributes = []
            for attr in {{input}}.columns:
                type_name = {{input}}.dtypes[attr]
                precision = None
                scale = None
                attributes.append({
                  'enumeration': 0,
                  'feature': 0,
                  'label': 0,
                  'name': attr,
                  'type': types_names[str(type_name)],
                  'nullable': True,
                  'metadata': None,
                  'precision': precision,
                  'scale': scale
                })
            parameters = {
                'name': "{{name}}",
                'is_first_line_header': write_header,
                'enabled': 1,
                'is_public': 0,
                'format': "{{format}}",
                'storage_id': {{storage}},
                'description': "{{description}}",
                'user_id': "{{user_id}}",
                'user_login': "{{user_login}}",
                'user_name': "{{user_name}}",
                'workflow_id': "{{workflow_id}}",
                'task_id': '{{task_id}}',
                'url': "{{final_url}}",
                'attributes': attributes
            }
            register_datasource('{{url}}', parameters, '{{token}}', 'overwrite')
            {%- endif %}
        """
        path = parsed.path

        ctx = dict(protect=protect,
                   path=path if not protect else os.path.basename(path),
                   hdfs_server=parsed.hostname,
                   hdfs_port=parsed.port,
                   hdfs_user=hdfs_user,
                   scheme=parsed.scheme,
                   name=self.name,
                   mode=self.mode,
                   storage=self.storage_id,
                   description=gettext(
                       'Data source generated by workflow {}').format(
                       self.workflow_id),
                   workflow_id=self.workflow_id,
                   format=self.format,
                   header=self.header,
                   user_name=self.user['name'],
                   user_id=self.user['id'],
                   user_login=self.user['login'],
                   tags=repr(self.tags),
                   FORMAT_CSV=self.FORMAT_CSV,
                   FORMAT_PICKLE=self.FORMAT_PICKLE,
                   FORMAT_JSON=self.FORMAT_JSON,
                   FORMAT_PARQUET=self.FORMAT_PARQUET,
                   data_types=json.dumps(self.PANDAS_TO_LIMONERO_DATA_TYPES),
                   final_url=final_url,
                   input=df_input,
                   token=token,
                   url=url,
                   error_file_exists=gettext('File already exists'),
                   warn_ignored=gettext('File not written (already exists)'),
                   error_invalid_mode=gettext(
                       'Invalid mode {}').format(self.mode),
                   uuid=uuid.uuid4().hex,
                   storage_url=storage['url'],
                   task_id=self.parameters['task_id'],
                   )

        return dedent(self.render_template(ctx))
