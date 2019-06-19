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
from sklearn.preprocessing import LabelBinarizer
from multiprocessing import Lock
# import threading
# import tensorflow as tf
# sess = tf.Session()

TarFileInfo = namedtuple('TarFileInfo',
                         'name, type, part_name, offset, size, '
                         'modified, permissions, owner, group, dir')

TarDirInfo = namedtuple('TarDirInfo',
                        'name, type, modified, permissions, owner, '
                        'group, files')

consumed_val = 0
consumed_train = 0
# consumed_val = tf.Variable(consumed_val)
#
# consumed_train = tf.Variable(consumed_train)
# lock_train = Lock()
# lock_val = Lock()

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
                 'shuffle', 'lock']

    def __init__(self, tar_path, batch_size=32, image_data_generator=None,
                 seed=None, split=None, shuffle=True, image_transformations={},
                 subset=None):
        self.lock = Lock()

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
        # self.tar.close()

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

    def read_tar(self, handler, file_path):
        tar_info = handler.getmember(file_path[1:])

        img = load_img(path=io.BytesIO(handler.extractfile(tar_info).read()),
                       **self.image_transformations
                       )
        img_array = img_to_array(img)

        return img_array

    def open_function(self):
        if self.tar_path.endswith('.gz') or self.tar_path.endswith('.tgz'):
            tar = tarfile.open(self.tar_path, 'r:gz')
        else:
            tar = tarfile.open(self.tar_path, 'r')
        return tar

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            yield archive_image_generator(reader_function=self.read_tar,
                                           open_function=self.open_function,
                                           files=self.dataset[self.subset],
                                           batch_size=self.batch_size,
                                           data_generator=self.image_data_generator,
                                           dir_list=self.dir_list,
                                           shuffle=self.shuffle,
                                           subset=self.subset)


def archive_image_generator(reader_function, open_function, files,
                            batch_size=32,
                            data_generator=None, dir_list=None, shuffle=True,
                            subset=None):
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
    global consumed_train, consumed_val, lock_val, lock_train

    filenames = [name for name, info in files.items() if info.type == 'file']
    if shuffle:
        random.shuffle(filenames)

    lenght_filenames = len(filenames)

    if subset == 'validation':
        while True:
            global consumed_val
            # lock_val.acquire()
            handler = open_function()
            labels = []
            images = []
            for name in filenames[consumed_val: consumed_val + batch_size]:
                parts = name.split('/')
                _cls = u'{}'.format('_'.join(parts[2:len(parts)-1]))
                labels.append(_cls)
                images.append(reader_function(handler, name))
            images = np.array(images)
            lb = LabelBinarizer()
            lb.fit(list(set(labels)))
            labels = lb.transform(labels)
            handler.close()

            return (images, labels)
            # print '=' * 20
            # print os.getpid(), subset, consumed_val, consumed_val + batch_size
            # print '=' * 20
            consumed_val += int(batch_size)
            if consumed_val >= lenght_filenames:
                consumed_val = 0
            # lock_val.release()
    else:
        while True:
            global consumed_train
            # lock_train.acquire()
            handler = open_function()
            labels = []
            images = []
            for name in filenames[consumed_train: consumed_train + batch_size]:
                parts = name.split('/')
                _cls = u'{}'.format('_'.join(parts[2:len(parts)-1]))
                labels.append(_cls)
                images.append(reader_function(handler, name))
            images = np.array(images)
            lb = LabelBinarizer()
            lb.fit(list(set(labels)))
            labels = lb.transform(labels)
            handler.close()

            return (images, labels)
            # print '=' * 20
            # print os.getpid(), subset, consumed_train, consumed_train + batch_size
            # print '=' * 20
            consumed_train += int(batch_size)
            if consumed_train >= lenght_filenames:
                consumed_train = 0
            # lock_train.release()


    # files_by_class = {}
    # for d in dir_list:
    #     files_by_class[d] = []
    #
    # total = 0
    # for info in files.values():
    #     if info.type == 'file':
    #         files_by_class[info.dir].append(info.name)
    #         total += 1
    #
    # if shuffle:
    #     for d in dir_list:
    #         random.shuffle(files_by_class[d])
    #
    # consumed = 0
    # number_of_classes = len(dir_list)
    #
    # while True:
    #     file_list = []
    #     tam = 0
    #     for cls in files_by_class:
    #         file_list += files_by_class[cls][consumed//number_of_classes:
    #                                          consumed//number_of_classes +
    #                                          batch_size//number_of_classes]
    #         #print '!'*20, cls, len(files_by_class[cls])
    #         tam += len(files_by_class[cls])
    #
    #     if shuffle:
    #         random.shuffle(file_list)
    #
    #     labels = []
    #     images = []
    #
    #     for name in file_list:
    #         parts = name.split('/')
    #         _cls = u'{}'.format('_'.join(parts[2:len(parts)-1]))
    #         labels.append(_cls)
    #         images.append(reader_function(name))
    #     images = np.array(images)
    #
    #     lb = LabelBinarizer()
    #     lb.fit(list(set(labels)))
    #     labels = lb.transform(labels)
    #
    #     # if the ImageDataGenerator object is not None, apply it
    #     print '#'*40
    #     print (subset, tam, '[', consumed, ':', consumed + batch_size, ']')
    #     print '#'*40
    #
    #     (images, labels) = data_generator.flow(images,
    #                                            labels,
    #                                            batch_size=batch_size
    #                                            ).next()
    #
    #     yield (images, labels)
    #     consumed += int(batch_size)


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

