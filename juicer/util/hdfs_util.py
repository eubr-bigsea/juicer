# coding=utf-8
import datetime
import os
import urllib
from collections import namedtuple

from urlparse import urlparse

import pyarrow as pa

HarInfo = namedtuple('HarInfo',
                     'name, type, part_name, offset, size, '
                     'modified, permissions, owner, group')


class HdfsHarFile(object):
    """
    Requires pyarrow, but it doesn't support HAR files natively. So, in this
    implementation, we read the index file and move the file pointer in order to
    retrieve files.
    """
    __slots__ = ['har_path', 'hdfs_url', 'file_list', 'loaded', '_server',
                 '_port']

    def __init__(self, hdfs_url, har_path):
        self.har_path = har_path
        self.hdfs_url = hdfs_url
        self.file_list = {}
        self.loaded = False

        parsed_url = urlparse(self.hdfs_url)
        self._server = parsed_url.server
        self._port = parsed_url.port

    def load(self):
        """
        Loads metadata information from _index file (files and directories
        listing).
        """
        fs = pa.hdfs.connect(self._server, self._port)
        index_name = os.path.join(self.har_path, '_index')

        self.file_list = []
        with fs.open(index_name, 'rb') as f:
            lines = f.read().decode('utf8').split('\n')
            for line in lines:
                cols = urllib.parse.unquote(line).split(' ')
                name, _type, part_name, offset, size, extra = cols
                offset = int(offset)
                size = int(size)

                modified, permissions, owner, group = extra.split('+')
                modified = datetime.datetime.fromtimestamp(
                    int(modified) * 0.001)

                self.file_list[name] = HarInfo(
                    name, _type, part_name, offset, size, modified, permissions,
                    owner, group)
        self.loaded = True

    def exists(self, path):
        return path in self.file_list

    def is_file(self, path):
        return self.exists(path) and self.file_list[path].type == 'file'

    def read(self, path):
        """
        Loads the content of a file from the HAR file.
        """
        if not self.is_file(path):
            raise ValueError(_('File does not exist in HAR: {}'.format(path)))

        part_name = os.path.join(self.har_path, path)
        file_info = self.file_list[path]

        fs = pa.hdfs.connect(self._server, self._port)
        with fs.open(part_name, 'rb') as f:
            f.seek(file_info.offset)
            data = f.read(file_info.size)
        return data
