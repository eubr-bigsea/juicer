# coding=utf-8
import io
import random
import tarfile
import tempfile
from collections import namedtuple, OrderedDict

import numpy as np
import os
from juicer.service import limonero_service
from juicer.util.hdfs_util import HdfsHarFileSystem, HdfsUtil
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

TarFileInfo = namedtuple('TarFileInfo',
                         'name, type, part_name, offset, size, '
                         'modified, permissions, owner, group')

TarDirInfo = namedtuple('TarDirInfo',
                        'name, type, modified, permissions, owner, '
                        'group, files')


def har_image_generator(hdfs_url, har_path, bs=32, mode='train', aug=None,
                        seed=None):
    """
    An image loader generator that reads images from HDFS stored in the HAR
    format.
    The structure in HAR file must follow the layout described in 
    archive_image_generator()
    :param har_path: Path for the HAR file
    :type har_path: str
    :param hdfs_url: hdfs_url URL for the HDFS server
    :type hdfs_url: str
    :param bs: The batch size
    :type bs: int
    :param seed: Random number generator seed
    :param mode: mode 'test' or 'train'
    :type mode: str
    :seed seed for random number generator
    :type seed: int
    :arg mode test of train
    :arg bs The batch size
    :param aug (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :type aug Augmentation
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
    :arg seed seed for random number generator
    """
    if tar_path.endswith('.gz'):
        tar = tarfile.open(tar_path, 'r:gz')
    else:
        tar = tarfile.open(tar_path, 'r')
    files = OrderedDict()
    for m in tar.getmembers():
        if m.isfile():
            files['/' + m.name] = TarFileInfo(
                '/' + m.name, 'file', None, None, m.size, m.mtime, None,
                m.uname, m.gname)
        elif m.isdir():
            files['/' + m.name] = TarDirInfo(
                '/' + m.name, 'dir', m.mtime, None, m.uname, m.gname, [])
        else:
            pass

    def read_tar(part_info, path):
        tar_info = tar.getmember(path[1:])
        img = load_img(io.BytesIO(tar.extractfile(tar_info).read()))
        img = img.resize((150, 150))
        img_array = img_to_array(img)
        # img_array = img_array.reshape((1,) + img_array.shape)

        return img_array

    return archive_image_generator(read_tar, files, bs, mode, aug, seed)


def archive_image_generator(read_fn, files, bs=32, mode='train', aug=None,
                            seed=None):
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
    :arg read_fn function to read to archive file
    :arg files dictionary with files information
    :arg mode test of train
    :arg bs The batch size
    :arg aug (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :arg seed seed for random number generator
    """
    if mode not in ['train', 'test']:
        raise ValueError(_('Invalid stage for generator: {}'.format(mode)))

    def files_in_mode(_info, _mode):
        return _info.type == 'file' and _info.name.startswith('/' + _mode + '/')

    file_list = [info.name for info in files.values() if
                 files_in_mode(info, mode)]
    if seed:
        random.seed(seed)
        random.shuffle(file_list)
    total = len(file_list)
    consumed = 1990
    while consumed < total:
        names = file_list[consumed:consumed + bs]
        labels = []
        images = []
        for name in names:
            parts = name.split('/', 3)
            labels.append(parts[2])
            info = files[name]
            images.append(read_fn(info.part_name, name))
        images = np.array(images)

        # if the data augmentation object is not None, apply it
        if aug is not None:
            (images, labels) = next(aug.flow(images,
                                             labels, batch_size=bs,
                                             save_to_dir='/tmp/preview'))

        yield (np.array(images), np.array(labels))
        consumed += int(bs)
        print(consumed)


def load_keras_model(limonero_url, token, storage_id, path):
    """
    Loads a Keras model with information provided by Limonero.
    :param limonero_url URL for Limonero
    :param token Limonero auth token
    :param storage_id Limonero storage id
    :param path Path where model will be stored
    :returns Loaded Keras model
    """
    storage = limonero_service.get_storage_info(limonero_url, token, storage_id)
    if storage.type not in ['HDFS', 'LOCAL']:
        raise ValueError(_('Unsupported storage type: {}'.format(storage.type)))

    final_path = os.path.join(storage.url, 'models', path)

    if storage.type == 'HDFS':
        # Stores the model in a temporary file to copy it from storage

        tmp_file, filename = tempfile.mkstemp()
        h = HdfsUtil(storage.url)
        # Requires temp file because Keras do not load from stream :(
        h.copy_to_local(final_path, filename)
        return load_model(filename)

    elif storage.type == 'LOCAL':
        return load_model(path)


def save_keras_model(limonero_url, token, storage_id, path, keras_model):
    """
    Saves a Keras model with information provided by Limonero.
    :param limonero_url URL for Limonero
    :param token Limonero auth token
    :param storage_id Limonero storage id
    :param path Path where model will be stored
    :param keras_model Model to be saved
    """
    storage = limonero_service.get_storage_info(limonero_url, token, storage_id)
    if storage.type not in ['HDFS', 'LOCAL']:
        raise ValueError(_('Unsupported storage type: {}'.format(storage.type)))

    if not path.endswith('.h5'):
        path += '.h5'
    # FIXME: review the path
    final_path = os.path.join(storage.url, 'models', path)

    if storage.type == 'HDFS':
        # Stores the model in a temporary file to copy it to storage
        tmp_file, filename = tempfile.mkstemp()
        keras_model.save(filename)

        # Copy to HDFS
        h = HdfsUtil(storage.url)
        h.copy_from_local(filename, final_path)
    elif storage.type == 'LOCAL':
        keras_model.save(path)


def main():
    url = 'hdfs://spark01.ctweb.inweb.org.br:9000'
    path = '/keras/dogs_cats.har'

    total = 0
    for block in har_image_generator(url, path):
        total += len(block)


def main1():
    path = '/var/tmp/dogs_cats.tar.gz'
    total = 0
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    for block in tar_image_generator(path, aug=data_gen):
        total += len(block)


if __name__ == '__main__':
    main1()
