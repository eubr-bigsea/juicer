# coding=utf-8
import io
import random
import tarfile
import tempfile
from collections import namedtuple, OrderedDict
from gettext import gettext

import numpy as np
import os
from juicer.service import limonero_service
from juicer.util.hdfs_util import HdfsHarFileSystem, HdfsUtil
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import numpy
import pandas
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

TarFileInfo = namedtuple('TarFileInfo',
                         'name, type, part_name, offset, size, '
                         'modified, permissions, owner, group, dir')

TarDirInfo = namedtuple('TarDirInfo',
                        'name, type, modified, permissions, owner, '
                        'group, files')


def har_image_generator(hdfs_url, har_path, batch_size=32, mode='train',
                        image_data_generator=None, seed=None):
    """
    An image loader generator that reads images from HDFS stored in the HAR
    format.
    The structure in HAR file must follow the layout described in 
    archive_image_generator()
    :param har_path: Path for the HAR file
    :type har_path: str
    :param hdfs_url: hdfs_url URL for the HDFS server
    :type hdfs_url: str
    :param batch_size: The batch size
    :type batch_size: int
    :param seed: Random number generator seed
    :param mode: mode 'test' or 'train'
    :type mode: str
    :seed seed for random number generator
    :type seed: int
    :arg mode test of train
    :arg batch_size The batch size
    :param image_data_generator (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :type image_data_generator Augmentation
    """
    h = HdfsHarFileSystem(hdfs_url, har_path)
    h.load()
    return archive_image_generator(h.read, h.file_list, batch_size, mode,
                                   image_data_generator, seed)


class ArchiveImageGenerator(object):
    """
    An image loader generator that reads images from TAR files (gzipped or not) 
    format.
    The structure in TAR file must follow the layout described in 
    archive_image_generator()
    :arg tar_path Path for the TAR file. If ends with .gz, it will be expanded.
    :arg bs The batch size
    :arg aug (default is None ) If an augmentation object is specified,
        then we’ll apply it before we yield our images and labels.
    :arg seed seed for random number generator
    """
    __slots__ = ['train', 'validation', 'files', 'batch_size',
                 'image_data_generator', 'image_transformations',
                 'tar', 'split', 'dataset', 'subset', 'tar_path', 'dir_list',
                 'shuffle']

    def __init__(self, tar_path, batch_size=32, image_data_generator=None,
                 seed=None, split=None, shuffle=True, image_transformations={},
                 subset=None):

        if subset is None:
            raise ValueError(
                gettext("It's necessary inform the subset parameter.")
            )

        supported_subset = ('training', 'validation')

        if subset not in supported_subset:
            raise ValueError(gettext('Unsupported subset type.'))

        if tar_path.endswith('.gz') or tar_path.endswith('.tgz'):
            self.tar = tarfile.open(tar_path, 'r:gz')
        else:
            self.tar = tarfile.open(tar_path, 'r')
        files = OrderedDict()
        self.dir_list = set()

        for m in self.tar.getmembers():
            name = '/{}'.format(m.name)
            if m.isfile():
                names = m.name.split('/')
                dir_file = '/'.join(names[:len(names)-1])
                self.dir_list.add(dir_file)

                files[name] = TarFileInfo(
                    name, 'file', None, None, m.size,
                    m.mtime, None, m.uname, m.gname, dir_file
                )
            elif m.isdir():
                files[name] = TarDirInfo(
                    name, 'dir', m.mtime, None, m.uname, m.gname, []
                )

        self.dir_list = list(self.dir_list)

        if seed:
            random.seed(seed)

        self.batch_size = batch_size
        self.image_data_generator = image_data_generator
        self.files = files
        self.image_transformations = image_transformations
        self.subset = subset
        self.tar_path = tar_path
        self.dataset = {}
        self.shuffle = shuffle

        if split is not None:
            part = int(len(files) * self.split)
            items = files.items()
            if self.subset == 'training':
                self.dataset[self.subset] = dict(items[:part])
            else:
                self.dataset[self.subset] = dict(items[part:])
        else:
            self.dataset[self.subset] = files

    def read_tar(self, file_path):
        tar_info = self.tar.getmember(file_path[1:])

        img = load_img(path=io.BytesIO(self.tar.extractfile(tar_info).read()),
                       **self.image_transformations
                       )
        img_array = img_to_array(img)

        return img_array

    def read(self):
        return archive_image_generator(reader_function=self.read_tar,
                                       files=self.dataset[self.subset],
                                       batch_size=self.batch_size,
                                       data_generator=self.image_data_generator,
                                       dir_list=self.dir_list,
                                       shuffle=self.shuffle)


def archive_image_generator(reader_function, files, batch_size=32,
                            data_generator=None, dir_list=[], shuffle=True):
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
    :arg reader_function function to read to archive file
    :arg files dictionary with files information
    :arg batch_size The batch size
    :arg data_generator (default is None ) If an ImageDataGenerator
         object is specified, then we’ll apply it before we yield our images
         and labels.
    """

    files_by_class = {}
    for d in dir_list:
        files_by_class[d] = []


    total = 0
    for info in files.values():
        if info.type == 'file':
            files_by_class[info.dir].append(info.name)
            total += 1

    if shuffle:
        for d in dir_list:
            random.shuffle(files_by_class[d])

    consumed = 0
    number_of_classes = len(dir_list)
    mapping = {}
    c = 0

    while consumed < total:
        file_list = []
        for cls in files_by_class:
            file_list += files_by_class[cls][consumed:
                                             consumed/number_of_classes +
                                             batch_size/number_of_classes]
        if shuffle:
            random.shuffle(file_list)

        labels = []
        images = []
        for name in file_list:
            parts = name.split('/')
            _cls = u'{}'.format('_'.join(parts[2:len(parts)-1]))
            labels.append(_cls)
            images.append(reader_function(name))
        images = np.array(images)

        print labels

        labels = np.asarray(labels)
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoded_labels = encoder.transform(labels)
        labels = np_utils.to_categorical(encoded_labels)

        images, labels = (np.array(images), np.array(labels))

        # if the ImageDataGenerator object is not None, apply it
        if data_generator is not None:
            (images, labels) = data_generator.flow(images,
                                                   labels,
                                                   batch_size=batch_size,
                                                   save_to_dir='/tmp/preview')\
                                                   .next()

        print (np.array(images), np.array(labels))
        yield (images, labels)
        consumed += int(batch_size)


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
        raise ValueError(
            gettext('Unsupported storage type: {}'.format(storage.type)))

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
        raise ValueError(
            gettext('Unsupported storage type: {}'.format(storage.type)))

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
    path = '/src/datadogs_cats_validation.tar.gz'
    total = 0
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_image_generator_train = ArchiveImageGenerator(
        tar_path='/src/datadogs_cats_train.tar.gz',
        batch_size=32,
        image_data_generator=data_gen,
        seed=None,
        image_transformations={'target_size': (256, 256), 'color_mode': u'rgb',
                               'interpolation': u'nearest'},
        subset='training'
    )
    train_image_generator_train = train_image_generator_train.read()
    print train_image_generator_train

    # for block in archive_image_generator(path, aug=data_gen):
    #     total += len(block)


if __name__ == '__main__':
    print "bla"
main1()
