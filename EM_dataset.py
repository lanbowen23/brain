from keras import backend as K
import h5py
import numpy as np
import random
import os.path
import scipy.ndimage as ndi
from keras.utils import Sequence


# row=256
# col=256
# depth=7
from numpy.core._multiarray_umath import ndarray


def random_transform(x, seed=None,
                     rotate=True, rotation_range=90,
                     h_flip=True, v_flip=True, z_flip=True):
    """Randomly augment a single image tensor.
    # Arguments
        x: 3D tensor, single image.
        seed: random seed.
    # Returns
        A randomly transformed version of the input (same shape).
    """
    # x y z when use in code
    # z x y for now

    img_row_axis = 1
    img_col_axis = 0
    img_channel_axis = 2

    if seed is not None:
        # print('seed: ', seed)
        np.random.seed(seed)

    # try to random choosing 3 kinds of rotation
    #     if rotation_range:
    #         theta = np.pi / 180 * \
    #                 np.random.uniform(-rotation_range, rotation_range)
    #     else:
    #         theta = 0

    if rotate:
        rand = np.random.random()
        if rand < 0.33:
            theta = np.pi / 180 * 90
        if rand > 0.67:
            theta = np.pi / 180 * 180
        else:
            theta = np.pi / 180 * 270

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode='nearest', cval=0)

    if h_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)

    if v_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_axis)

    if z_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_channel_axis)

    return x

class EM_Sequence(Sequence):

    def __init__(self, batch_size, patch_shape, 
                 train_flag=True, data_dir='DATA/', data_file='full_c3.h5'):
        assert (len(patch_shape) == 3)
        self.patch_shape = patch_shape  # 256 x 256 x 7
        self.data_dir = data_dir
        self.data_file = data_file
        d_file = self.data_dir + self.data_file
        h5f = h5py.File(d_file, 'r')
        print('loading data file from: {0}'.format(d_file))
        self.data = h5f['data']
        self.label = h5f['label']
        self.image_size = self.data.shape
        self.batch_size = batch_size
        self.train_flag = train_flag

    def __len__(self):
        image_size = np.prod(self.image_size)
        patch_size = np.prod(self.patch_shape)
        return int(np.ceil(image_size / patch_size) * 16)

    def __getitem__(self, idx):
        x_patch, y_patch, z_patch = self.patch_shape
        x_input, y_input, z_input= self.data.shape

        # print("size of batch: {0}".format(batch_size))
        # print("size of zpatch: {0}".format(z_patch))

        batch_x = np.zeros(tuple([self.batch_size] + [x_patch, y_patch, z_patch]), dtype=K.floatx())
        batch_y = np.zeros(tuple([self.batch_size] + [x_patch, y_patch, z_patch]), dtype=K.floatx())
        split_slice = 25
        for i in range(self.batch_size):
            x_start = random.randrange(0, x_input - x_patch)
            x_end = x_start + x_patch
            y_start = random.randrange(0, y_input - y_patch)
            y_end = y_start + y_patch
            if self.train_flag:
                z_start = random.randrange(split_slice, z_input - z_patch)
                z_end = z_start + z_patch
            else:
                z_start = random.randrange(0, split_slice - z_patch)
                z_end = z_start + z_patch
            X = self.data[x_start:x_end, y_start:y_end, z_start:z_end]
            Y = self.label[x_start:x_end, y_start:y_end, z_start:z_end]
            # do the data augmentation here use the same seed for X and Y
            # idx is empty for the start of validation
            seed = np.random.randint(17)
            X = random_transform(X, seed=seed)
            Y = random_transform(Y, seed=seed)
            batch_x[i] = X
            batch_y[i] = Y

        batch_x = np.transpose(batch_x, (0, 3, 1, 2))
        batch_y = np.transpose(batch_y, (0, 3, 1, 2))
        # sample, time, channel, width, height
        batch_x = batch_x.reshape(self.batch_size, z_patch, 1, x_patch, y_patch)
        batch_y = batch_y.reshape(self.batch_size, z_patch, 1, x_patch, y_patch)
        batch_y_r = 1 - batch_y
        # -- notice that label is converted to 2 channel representing
        # binary categorical labels so the softmax function/ layer
        # can use it to element-wisely compute SUM(y*log(y_hat))
        # loss across pixels in the predicted probability map.
        # Similarly used in train_generator_5D
        batch_y = np.concatenate((batch_y, batch_y_r), axis=2)

        return np.array(batch_x), np.array(batch_y)

class EM_Data:

    def __init__(self, patch_shape, data_dir='DATA/', data_file='full_c3.h5'):
        assert (len(patch_shape) == 3)
        self.patch_shape = patch_shape  # 256 x 256 x 7
        self.valid_data = None

        self.data_dir = data_dir
        self.data_file = data_file

        d_file = self.data_dir + self.data_file
        h5f = h5py.File(d_file, 'r')
        print('loading data file from: {0}'.format(d_file))
        self.data = h5f['data']
        self.label = h5f['label']

        self.image_size = self.data.shape

    def train_generator_5D(self):
        data_dir = self.data_dir
        data_files = ('snemi3d_train_full_stacks_v1.h5', 'snemi3d_train_full_stacks_v2.h5',
                      'snemi3d_train_full_stacks_v3.h5', 'snemi3d_train_full_stacks_v4.h5',
                      'snemi3d_train_full_stacks_v5.h5', 'snemi3d_train_full_stacks_v6.h5',
                      'snemi3d_train_full_stacks_v7.h5', 'snemi3d_train_full_stacks_v8.h5',
                      'snemi3d_train_full_stacks_v9.h5', 'snemi3d_train_full_stacks_v10.h5',
                      'snemi3d_train_full_stacks_v11.h5', 'snemi3d_train_full_stacks_v12.h5',
                      'snemi3d_train_full_stacks_v13.h5', 'snemi3d_train_full_stacks_v14.h5',
                      'snemi3d_train_full_stacks_v15.h5', 'snemi3d_train_full_stacks_v16.h5')

        d_files = []
        h5fs = []
        for i in range(len(data_files)):  # 16 files
            d_files.append(data_dir + data_files[i])
        self.data_all = []
        self.label_all = []
        for i in range(len(d_files)):
            h5fs.append(h5py.File(d_files[i], 'r'))
            self.data_all.append(h5fs[i]['data'])  # 16 x 100
            self.label_all.append(h5fs[i]['label'])

        x_patch = self.patch_shape[0]
        y_patch = self.patch_shape[1]
        z_patch = self.patch_shape[2]

        x_input = self.data.shape[0]
        y_input = self.data.shape[1]
        z_input = self.data.shape[2]

        print("size of zpatch{0}".format(z_patch))
        n_files = len(self.data_all)
        start_slice = 20
        while True:
            file_id = random.randrange(0, n_files)
            x_start = random.randrange(0, x_input - x_patch)
            x_end = x_start + x_patch
            y_start = random.randrange(0, y_input - y_patch)
            y_end = y_start + y_patch
            z_start = random.randrange(start_slice, z_input - z_patch)
            z_end = z_start + z_patch
            X = self.data_all[file_id][x_start:x_end, y_start:y_end, z_start:z_end]
            Y = self.label_all[file_id][x_start:x_end, y_start:y_end, z_start:z_end]

            X = np.transpose(X, (2, 0, 1))
            Y = np.transpose(Y, (2, 0, 1))
            # sample, time, channel, width, height
            X_sec = X.reshape(1, z_patch, 1, x_patch, y_patch)
            Y_0 = Y.reshape(1, z_patch, 1, x_patch, y_patch)
            Y_1 = 1 - Y_0
            Y_sec = np.concatenate((Y_0, Y_1), axis=2)  # concatenate on channel axis
            yield X_sec, Y_sec

    def train_generator_aug(self, batch_size=32, start_slice=20):
        x_patch = self.patch_shape[0]
        y_patch = self.patch_shape[1]
        z_patch = self.patch_shape[2]

        x_input = self.data.shape[0]
        y_input = self.data.shape[1]
        z_input = self.data.shape[2]

        # print("size of batch: {0}".format(batch_size))
        # print("size of zpatch: {0}".format(z_patch))

        start_slice = start_slice
        while True:
            batch_x = np.zeros(tuple([batch_size] + [x_patch, y_patch, z_patch]), dtype=K.floatx())
            batch_y = np.zeros(tuple([batch_size] + [x_patch, y_patch, z_patch]), dtype=K.floatx())

            for i in range(batch_size):
                x_start = random.randrange(0, x_input - x_patch)
                x_end = x_start + x_patch
                y_start = random.randrange(0, y_input - y_patch)
                y_end = y_start + y_patch
                z_start = random.randrange(start_slice, z_input - z_patch)
                z_end = z_start + z_patch

                X = self.data[x_start:x_end, y_start:y_end, z_start:z_end]
                Y = self.label[x_start:x_end, y_start:y_end, z_start:z_end]

                # do the data augmentation here use the same seed for X and Y
                seed = np.random.randint(100)
                X = random_transform(X, seed=seed)
                Y = random_transform(Y, seed=seed)

                batch_x[i] = X
                batch_y[i] = Y

            batch_x = np.transpose(batch_x, (0, 3, 1, 2))
            batch_y = np.transpose(batch_y, (0, 3, 1, 2))
            # sample, time, channel, width, height
            batch_x = batch_x.reshape(batch_size, z_patch, 1, x_patch, y_patch)
            batch_y = batch_y.reshape(batch_size, z_patch, 1, x_patch, y_patch)
            batch_y_r = 1 - batch_y
            # -- notice that label is converted to 2 channel representing
            # binary categorical labels so the softmax function/ layer
            # can use it to element-wisely compute SUM(y*log(y_hat))
            # loss across pixels in the predicted probability map.
            # Similarly used in train_generator_5D
            batch_y = np.concatenate((batch_y, batch_y_r), axis=2)
            yield batch_x, batch_y

    def valid_generator_5D(self, batch_size=2, slice_end=20):
        # data_dir = 'DATA/augment/'
        # data_files = ('snemi3d_train_full_stacks_v1.h5', 'snemi3d_train_full_stacks_v2.h5')
        data_dir = self.data_dir
        data_files = self.data_file
        d_files = data_dir + data_files
        h5fs = h5py.File(d_files, 'r')
        self.valid_data_all = h5fs['data']
        self.valid_label_all = h5fs['label']

        x_patch = self.patch_shape[0]
        y_patch = self.patch_shape[1]
        z_patch = self.patch_shape[2]

        x_input = self.data.shape[0]
        y_input = self.data.shape[1]
        z_input = self.data.shape[2]

        slice_end = slice_end
        # n_files = len(self.valid_data_all)
        while True:
            batch_x = np.zeros(tuple([batch_size] + [x_patch, y_patch, z_patch]), dtype=K.floatx())
            batch_y = np.zeros(tuple([batch_size] + [x_patch, y_patch, z_patch]), dtype=K.floatx())

            for i in range(batch_size):
                x_start = random.randrange(0, x_input - x_patch)
                x_end = x_start + x_patch
                y_start = random.randrange(0, y_input - y_patch)
                y_end = y_start + y_patch
                z_start = random.randrange(0, slice_end - z_patch)
                z_end = z_start + z_patch
                # file_id = random.randrange(0, n_files)

                X = self.valid_data_all[x_start:x_end, y_start:y_end, z_start:z_end]
                Y = self.valid_label_all[x_start:x_end, y_start:y_end, z_start:z_end]

                batch_x[i] = X
                batch_y[i] = Y

            batch_x = np.transpose(batch_x, (0, 3, 1, 2))
            batch_y = np.transpose(batch_y, (0, 3, 1, 2))
            # sample, time, channel, width, height
            batch_x = batch_x.reshape(batch_size, z_patch, 1, x_patch, y_patch)
            batch_y_0 = batch_y.reshape(batch_size, z_patch, 1, x_patch, y_patch)
            batch_y_1 = 1 - batch_y_0
            batch_y = np.concatenate((batch_y_0, batch_y_1), axis=2)
            yield batch_x, batch_y

    def load_test_data_5D(self, test_file, depth):
        data_dir = self.data_dir
        data_file = test_file
        # data_file='snemi3d_test_v1.h5'
        if self.valid_data is None:
            d_file = data_dir + data_file
            h5f = h5py.File(d_file, 'r')
            self.valid_data = h5f['data']

        x_size = self.image_size[0]
        y_size = self.image_size[1]
        z_size = self.image_size[2]
        slices = depth
        x_start = 0
        y_start = 0
        z_start = 0
    
        x_end = x_start + x_size
        y_end = y_start + y_size
        z_end = z_start + depth
        X_test = np.ndarray((1, slices, 1, x_size, y_size), dtype=np.float32)
        X_test = self.valid_data[x_start:x_end, y_start:y_end, z_start:z_end]
        X_test = X_test.reshape(1, 1, x_size, y_size, slices)
        X_test = np.transpose(X_test, (0, 4, 1, 2, 3))
    
        return X_test

    def load_valid_data_5D(self, start_slice=0, slices=20):
        data_dir = self.data_dir
        data_file = self.data_file
        d_file = data_dir + data_file
        h5f = h5py.File(d_file, 'r')
        data = h5f['data']
        label = h5f['label']

        row = self.image_size[0]
        col = self.image_size[1]
        depth = slices

        x_size = row
        y_size = col
        slices = depth  # 20

        x_start = 0
        y_start = 0
        z_start = start_slice

        x_end = x_start + x_size
        y_end = y_start + y_size
        z_end = z_start + depth

        X_test = np.ndarray((1, slices, 1, x_size, y_size), dtype=np.float32)
        X_test = data[x_start:x_end, y_start:y_end, z_start:z_end]  # .reshape(1,1,x_size, y_size)
        X_test = X_test.reshape(1, 1, x_size, y_size, depth)
        X_test = np.transpose(X_test, (0, 4, 1, 2, 3))

        Y_test = np.ndarray((1, slices, 1, x_size, y_size), dtype=np.float32)
        Y_test = label[x_start:x_end, y_start:y_end, z_start:z_end]  # .reshape(1,1,x_size, y_size)
        Y_test = Y_test.reshape(1, 1, x_size, y_size, depth)
        Y_test = np.transpose(Y_test, (0, 4, 1, 2, 3))

        return X_test, Y_test


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=3,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
