# coding=utf-8
import urllib
from collections import namedtuple, OrderedDict
from urlparse import urlparse

import datetime

import os
import pyarrow as pa

HarFileInfo = namedtuple('HarFileInfo',
                         'name, type, part_name, offset, size, '
                         'modified, permissions, owner, group')

HarDirInfo = namedtuple('HarDirInfo',
                        'name, type, modified, permissions, owner, '
                        'group, files')


class HdfsUtil(object):
    __slots__ = ['hdfs_url', '_server', '_port']

    def __init__(self, hdfs_url):
        self.hdfs_url = hdfs_url

        parsed_url = urlparse(self.hdfs_url)
        self._server = parsed_url.hostname
        self._port = parsed_url.port

    def copy_from_local(self, local_path, hdfs_path):
        fs = pa.hdfs.connect(self._server, self._port)
        with open(local_path) as f:
            fs.upload(hdfs_path, f)

    def copy_to_local(self, local_path, hdfs_path):
        fs = pa.hdfs.connect(self._server, self._port)
        with open(local_path, 'wb') as f:
            fs.download(hdfs_path, f)

    def read(self, hdfs_path):
        fs = pa.hdfs.connect(self._server, self._port)
        with fs.open(hdfs_path, 'rb') as f:
            return f.read()


class HdfsHarFileSystem(object):
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
        self.file_list = OrderedDict()
        self.loaded = False

        parsed_url = urlparse(self.hdfs_url)
        self._server = parsed_url.hostname
        self._port = parsed_url.port

    def load(self):
        """
        Loads metadata information from _index file (files and directories
        listing).
        """
        fs = pa.hdfs.connect(self._server, self._port)
        index_name = os.path.join(self.har_path, '_index')

        self.file_list = OrderedDict()
        with fs.open(index_name, 'rb') as f:
            lines = f.read().decode('utf8').strip().split('\n')
            for line in lines:
                cols = urllib.unquote(line).strip().split(' ')
                if cols[1] == 'dir':
                    name, _type, extra = cols[:3]
                    modified, permissions, owner, group = extra.split('+')
                    # Ignore first information about size zero
                    files = cols[(len(cols) - 3) // 2 + 3:]
                    self.file_list[name] = HarDirInfo(
                        name, _type, modified, permissions, owner, group, files)
                else:
                    name, _type, part_name, offset, size, extra = cols
                    offset = int(offset)
                    size = int(size)

                    modified, permissions, owner, group = extra.split('+')
                    modified = datetime.datetime.fromtimestamp(
                        int(modified) * 0.001)

                    self.file_list[name] = HarFileInfo(
                        name, _type, part_name, offset, size, modified,
                        permissions, owner, group)
        self.loaded = True

    def exists(self, path):
        return path in self.file_list

    def is_file(self, path):
        return self.exists(path) and self.file_list[path].type == 'file'

    def is_dir(self, path):
        return self.exists(path) and self.file_list[path].type == 'dir'

    def read(self, part, path):
        """
        Loads the content of a file from the HAR file.
        """
        if not self.is_file(path):
            raise ValueError(_('File does not exist in HAR: {}'.format(path)))

        part_name = os.path.join(self.har_path, part)
        file_info = self.file_list[path]

        fs = pa.hdfs.connect(self._server, self._port)
        with fs.open(part_name, 'rb') as f:
            f.seek(file_info.offset)
            data = f.read(file_info.size)
        return data

    def ls(self, path):
        """
        Lists directory content information or file information;
        """
        if not self.exists(path):
            raise ValueError(_('File does not exist in HAR: {}'.format(path)))

        file_info = self.file_list[path]
        if file_info.type == 'dir':
            # return [self.file_list[os.path.join(path, name)] for name in
            #         file_info.files]
            return [os.path.join(path, name) for name in file_info.files]
        else:
            return file_info.name
