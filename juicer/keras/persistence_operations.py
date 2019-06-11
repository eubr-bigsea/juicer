# coding=utf-8
import random
import numpy as np
import os
import tarfile
from juicer.util.hdfs_util import HdfsHarFileSystem

from collections import namedtuple, OrderedDict 
TarFileInfo = namedtuple('TarFileInfo',
                         'name, type, part_name, offset, size, '
                         'modified, permissions, owner, group')

TarDirInfo = namedtuple('TarDirInfo',
                        'name, type, modified, permissions, owner, '
                        'group, files')


def har_image_generator(hdfs_url, har_path, bs=32, mode='train', aug=None, seed=None):
    """
    An image loader generator that reads images from HDFS stored in the HAR
    format.
    The structure in HAR file must follow the layout described in 
    archive_image_generator()
    :arg hdfs_url URL for the HDFS server
    :arg har_path Path for the HAR file
    :arg mode test of train
    :arg bs The batch size
    :arg aug (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :seed seed for random number generator
    """
    h = HdfsHarFileSystem(hdfs_url, har_path)
    h.load()
    return archive_image_generator(h.read, h.file_list, bs, mode, aug, seed)


def tar_image_generator(tar_path, bs=32, mode='train', aug=None, seed=None):
    """
    An image loader generator that reads images from TAR files (gzipped or not) 
    format.
    The structure in TAR file must follow the layout described in 
    archive_image_generator()
    :arg tar_path Path for the TAR file. If ends with .gz, it will be expanded.
    :arg mode test of train
    :arg bs The batch size
    :arg aug (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :seed seed for random number generator
    """
    if tar_path.endswith('.gz'):
        tar = tarfile.open(tar_path, 'r:gz')
    else:
        tar = tarfile.open(tar_path, 'r')
    files = OrderedDict()
    for m in tar.getmembers():
        if m.isfile():
            files['/' + m.name] = TarFileInfo('/' + m.name, 'file', None, None, m.size, 
                    m.mtime, None, m.uname, m.gname)
        elif m.isdir():
            files['/' + m.name] = TarDirInfo('/' + m.name, 'dir',  
                    m.mtime, None, m.uname, m.gname, [])
        else:
            pass
    def read_tar(part_info, path):
        tar_info = tar.getmember(path[1:])
        return tar.extractfile(tar_info)

    return archive_image_generator(read_tar, files, bs, mode, aug, seed)


def archive_image_generator(read_fn, files, bs=32, mode='train', aug=None, seed=None):
    """
    An image loader generator that reads a archive file format.
    The structure in file must follow the layout:
     /
    ├── train/
    │   ├── class1
    │   |   ├── image1.jpg
    │   |   ├── image2.jpg
    │   |   ├── ...
    │   |   ├── imageN.jpg
    │   ├── class2
    │   |   ├── image1.jpg
    │   |   ├── image2.jpg
    │   |   ├── ...
    │   |   ├── imageN.jpg
    │   └── class3
    │   |   ├── image1.jpg
    │   |   ├── image2.jpg
    │   |   ├── ...
    │   |   ├── imageN.jpg
    ├── test/
    │   ├── class1
    │   |   ├── image1.jpg
    │   |   ├── image2.jpg
    │   |   ├── ...
    │   |   ├── imageK.jpg
    │   ├── class2
    │   |   ├── image1.jpg
    │   |   ├── image2.jpg
    │   |   ├── ...
    │   |   ├── imageK.jpg
    │   └── class3
    │   |   ├── image1.jpg
    │   |   ├── image2.jpg
    │   |   ├── ...
    │   |   ├── imageK.jpg
    :arg file_handle file handle pointing to archive file
    :arg mode test of train
    :arg bs The batch size
    :arg aug (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :seed seed for random number generator
    """
    if mode not in ['train', 'test']:
        raise ValueError(_('Invalid stage for generator: {}'.format(mode)))

    def files_in_mode(info, mode):
        return info.type == 'file' and info.name.startswith('/' + mode + '/') 

    file_list = [info.name for info in files.values() if files_in_mode(info, mode)]
    if seed:
        random.seed(SEED)
        random.shuffle(file_list)
    total = len(file_list)
    consumed = 0
    print total
    while consumed < total:
        names = file_list[consumed:consumed + bs]
        labels = []
        images = []
        for name in names:
            parts = name.split('/', 3)
            labels.append(parts[2])
            info = files[name]        
            images.append(read_fn(info.part_name, name))
        # if the data augmentation object is not None, apply it
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images),
                labels, batch_size=bs))

        yield (np.array(images), np.array(labels))
        consumed += int(bs)
        print(consumed)

def main():
    url = 'hdfs://spark01.ctweb.inweb.org.br:9000'
    path = '/keras/dogs_cats.har'

    total = 0
    for block in har_image_generator(url, path):
        total += len(block)

def main1():
    path = '/var/tmp/dogs_cats.tar.gz'
    total = 0
    for block in tar_image_generator(path):
        total += len(block)

if __name__ == '__main__':
    main1()
