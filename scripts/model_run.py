# import os
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import animation
from torchvision.utils import *
import scipy

from tensorboardX import SummaryWriter

# from skimage import io, transform
from torch.utils.data import Dataset
import torchvision

import os
import time
import random
import scipy.misc


def show(img):
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


# print(os.getcwd())
# os.mkdir('./DebugOutput')

def save_as_image(a, img_name='test', img_dir='./DebugOutput'):
    N, C, H, W = a.shape

    a = np.transpose(a, (2, 3, 1, 0))

    for n in range(N):
        filename = os.path.join(img_dir, img_name + str(n) + '.tif')
        if C == 3:
            plt.imsave(filename, a[:, :, :, n])
            continue
        if C == 1:
            plt.imsave(filename, a[:, :, 0, n])
            continue
        pass


# import hyper_search
# import plot_run
# from deep_learning.analysis import utils

# dir(utils)

# %% md

## Load image data

# %%

img_dir = ['/mnt/disk1/TrainingData/Perf_SAX_SEG',
           '/mnt/disk1/TrainingData/Perf_SAX_SEG_Set2']


# %% md

## Data augmentation

# %%

class RandomFlip1stDim(object):
    """Randomly flip the first dimension of numpy array.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img ([N RO E1 ... ]): Image to be flipped.
        Returns:
            res: Randomly flipped image.
        """
        # print(img[0].shape)
        # print(img[1].shape)

        if random.random() < self.p:
            a = np.transpose(img[0], [1, 2, 0])
            a = np.flipud(a)
            a = np.transpose(a, [2, 0, 1])

            b = np.transpose(img[1], [1, 2, 0])
            b = np.flipud(b)
            b = np.transpose(b, [2, 0, 1])
            return (a.copy(), b.copy(), img[2])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomFlip2ndDim(object):
    """Randomly flip the second dimension of numpy array.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img ([N RO E1 ... ]): Image to be flipped.
        Returns:
            res: Randomly flipped image.
        """
        if random.random() < self.p:
            a = np.transpose(img[0], [1, 2, 0])
            a = np.fliplr(a)
            a = np.transpose(a, [2, 0, 1])

            b = np.transpose(img[1], [1, 2, 0])
            b = np.fliplr(b)
            b = np.transpose(b, [2, 0, 1])
            return (a.copy(), b.copy(), img[2])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomPermute2DT(object):
    """Randomly permute 1st and 2nd dimensions of numpy array.
    Args:
        p (float): probability of the image being permuted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img ([N RO E1 ... ]): Image to be flipped.
        Returns:
            res: Randomly flipped image.
        """
        if random.random() < self.p:
            return (np.transpose(img[0], (0, 2, 1)).copy(),
                    np.transpose(img[1], (0, 2, 1)).copy(), img[2])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop2DT(object):
    """Randomly crop the numpy array, fir 2D+T.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, roi, p=0.5, ro_range=(-24, 24), e1_range=(-24, 24),
        t_range=(-6, 6)):
        self.p = p
        self.ro_range = ro_range
        self.e1_range = e1_range
        self.t_range = t_range

    def __call__(self, img):
        """
        Args:
            img ([Ro E1 N ... ]): Image to be cropped.
        Returns:
            res: Randomly cropped image.
        """
        if random.random() < self.p:

            RO, E1, N = img[0].shape

            roi = img[2]

            ps_x = roi[0].astype(int)
            pe_x = roi[1].astype(int)
            ps_y = roi[2].astype(int)
            pe_y = roi[3].astype(int)
            aif_s = roi[4].astype(int)
            aif_e = roi[5].astype(int)

            ro_shifts = np.random.randint(self.ro_range[0],
                                          self.ro_range[1] + 1, 1)
            e1_shifts = np.random.randint(self.e1_range[0],
                                          self.e1_range[1] + 1, 1)
            t_shifts = np.random.randint(self.t_range[0], self.t_range[1] + 1,
                                         1)

            ss_ps_x = ps_x + ro_shifts
            ss_ps_y = ps_y + e1_shifts
            ss_ps_t = aif_s + t_shifts

            ss_pe_x = pe_x + ro_shifts
            ss_pe_y = pe_y + e1_shifts
            ss_pe_t = aif_e + t_shifts

            if (ss_ps_x < 0 or ss_ps_y < 0 or ss_ps_t < 0):
                return img

            if (ss_pe_x >= RO and ss_pe_y >= E1 and ss_pe_t >= N):
                return img

            a = img[0][ss_ps_t:ss_pe_t, ss_ps_x:ss_pe_x, ss_ps_y:ss_pe_y]
            b = np.expand_dims(img[1][0, ss_ps_x:ss_pe_x, ss_ps_y:ss_pe_y],
                               axis=0)

            return (a, b, img[2])

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    # %% md


## Load data and apply random crop

# %%

import scipy.io


class PerfDatasetRandomCrop(Dataset):
    """Perfusion dataset."""

    def __init__(self, img_dir, which_mask='myo', num_of_random=9,
        transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.which_mask = which_mask  # myo or endo or epi
        self.num_of_random = num_of_random

        self.ro_range = (-24, 24)
        self.e1_range = (-24, 24)
        self.t_range = (-6, 6)

        # find all images
        a = []
        for case_dir in self.img_dir:
            locations = os.listdir(case_dir)
            for loc in locations:
                if (os.path.isdir(os.path.join(case_dir, loc))):
                    a.extend(os.listdir(os.path.join(case_dir, loc)))

        num_samples = len(a)
        print("Found %d cases ... " % num_samples)

        self.Gd = []
        self.endo_masks = []
        self.epi_masks = []
        self.myo_masks = []
        self.endo_epi_masks = []
        self.endo_epi_rvi_masks = []
        self.endo_epi_rv_masks = []
        self.endo_epi_rv_rvi_masks = []
        self.rvi_pt = []
        self.names = []

        t0 = time.time()
        print("Start loading cases ... ")

        total_case_loaded = 0
        total_num_loaded = 0

        for case_dir in self.img_dir:
            locations = os.listdir(case_dir)
            for loc in locations:
                a = os.listdir(os.path.join(case_dir, loc))
                print('---> Start loading ', case_dir, loc)
                for ii, n in enumerate(a):

                    # if (ii>5):
                    #    break

                    print('------> Start loading %d out of %d, %s' % (
                        total_case_loaded, num_samples, n))
                    name = os.path.join(loc, n)

                    is_seg_norm = True

                    try:
                        mat = scipy.io.loadmat(
                            os.path.join(case_dir, name, 'Seg_norm.mat'))
                    except:
                        mat = scipy.io.loadmat(
                            os.path.join(case_dir, name, 'Seg.mat'))
                        is_seg_norm = False

                    Seg = mat['Seg']
                    num_seg = len(Seg[0])

                    Gd_all = self.load_one_data(case_dir, name,
                                                'Gd_resized_norm')
                    roi_all = self.load_one_data(case_dir, name, 'roi')

                    total_case_loaded += 1

                    for i in np.arange(num_seg):

                        Gd = Gd_all[:, :, :, i]

                        endo, epi, myo, endo_epi, endo_epi_rv, endo_epi_rv_rvi, endo_epi_rvi, rvi_pt, roi = self.load_from_Seg(
                            Seg, i, is_seg_norm)
                        Gd, endo, epi, myo, endo_epi, endo_epi_rv, endo_epi_rv_rvi, endo_epi_rvi, rvi_pt = self.load_from_numpy_array(
                            Gd, endo, epi, myo, endo_epi, endo_epi_rv,
                            endo_epi_rv_rvi, endo_epi_rvi, rvi_pt)

                        # roi = roi_all.flatten()

                        roi = roi.flatten()

                        N, RO, E1 = Gd.shape

                        ro_shifts = np.random.randint(self.ro_range[0],
                                                      self.ro_range[1] + 1,
                                                      self.num_of_random)
                        e1_shifts = np.random.randint(self.e1_range[0],
                                                      self.e1_range[1] + 1,
                                                      self.num_of_random)
                        t_shifts = np.random.randint(self.t_range[0],
                                                     self.t_range[1] + 1,
                                                     self.num_of_random)

                        ps_x = roi[0].astype(int)
                        pe_x = roi[1].astype(int)
                        ps_y = roi[2].astype(int)
                        pe_y = roi[3].astype(int)
                        aif_s = roi[4].astype(int)
                        aif_e = roi[5].astype(int)

                        if (i == 0):
                            print(
                                '    ro, [start, end] = %d, %d; e1, [start, end] = %d, %d; t, [start, end] = %d, %d' % (
                                    ps_x, pe_x, ps_y, pe_y, aif_s, aif_e))

                        # print('    Gd = %f, endo = %f, epi = %f' % (np.linalg.norm(Gd), np.linalg.norm(endo), np.linalg.norm(epi)))

                        # random crop
                        for rc in np.arange(self.num_of_random + 1):

                            if (rc == self.num_of_random):
                                ss_ps_x = ps_x;
                                ss_ps_y = ps_y;
                                ss_ps_t = aif_s;

                                ss_pe_x = pe_x;
                                ss_pe_y = pe_y;
                                ss_pe_t = aif_e;
                            else:
                                ss_ps_x = ps_x + ro_shifts[rc];
                                ss_ps_y = ps_y + e1_shifts[rc];
                                ss_ps_t = aif_s + t_shifts[rc];

                                ss_pe_x = pe_x + ro_shifts[rc];
                                ss_pe_y = pe_y + e1_shifts[rc];
                                ss_pe_t = aif_e + t_shifts[rc];

                            if (ss_ps_t < 0):
                                ss_ps_t = 0
                                ss_pe_t = 48

                            if (ss_pe_t > N):
                                ss_pe_t = N
                                ss_ps_t = ss_pe_t - 48

                            if (ss_ps_x < 0 or ss_ps_y < 0 or ss_ps_t < 0):
                                continue;

                            if (ss_pe_x > RO and ss_pe_y > E1 and ss_pe_t > N):
                                continue;

                            Gd_s = Gd[ss_ps_t:ss_pe_t, ss_ps_x:ss_pe_x,
                                   ss_ps_y:ss_pe_y]
                            endo_s = np.expand_dims(
                                endo[0, ss_ps_x:ss_pe_x, ss_ps_y:ss_pe_y],
                                axis=0)
                            epi_s = np.expand_dims(
                                epi[0, ss_ps_x:ss_pe_x, ss_ps_y:ss_pe_y],
                                axis=0)
                            myo_s = np.expand_dims(
                                myo[0, ss_ps_x:ss_pe_x, ss_ps_y:ss_pe_y],
                                axis=0)
                            endo_epi_s = np.expand_dims(
                                endo_epi[0, ss_ps_x:ss_pe_x, ss_ps_y:ss_pe_y],
                                axis=0)
                            endo_epi_rv_s = np.expand_dims(
                                endo_epi_rv[0, ss_ps_x:ss_pe_x,
                                ss_ps_y:ss_pe_y], axis=0)
                            endo_epi_rvi_s = np.expand_dims(
                                endo_epi_rvi[0, ss_ps_x:ss_pe_x,
                                ss_ps_y:ss_pe_y], axis=0)
                            endo_epi_rv_rvi_s = np.expand_dims(
                                endo_epi_rv_rvi[0, ss_ps_x:ss_pe_x,
                                ss_ps_y:ss_pe_y], axis=0)

                            Gd_s = Gd_s / np.max(Gd_s)

                            if (Gd_s.shape[0] != 48):
                                continue;
                            if (Gd_s.shape[1] != 176):
                                continue;
                            if (Gd_s.shape[2] != 176):
                                continue;

                            self.Gd.append(Gd_s)
                            self.endo_masks.append(endo_s)
                            self.epi_masks.append(epi_s)
                            self.myo_masks.append(myo_s)
                            self.endo_epi_masks.append(endo_epi_s)
                            self.endo_epi_rv_masks.append(endo_epi_rv_s)
                            self.endo_epi_rv_rvi_masks.append(endo_epi_rv_rvi_s)
                            self.endo_epi_rvi_masks.append(endo_epi_rvi_s)
                            self.rvi_pt.append(rvi_pt)

                            self.names.append(name + '_' + str(i))

                    total_num_loaded += (self.num_of_random + 1)

                    t1 = time.time()
                    print(
                        "             Time from starting : %f seconds ... \n" % (
                            t1 - t0))

                    # if total_num_loaded%100 == 0 and ii>0:
                    #    print("load %d " % total_num_loaded)

    def __len__(self):
        return len(self.Gd)

    def __getitem__(self, idx):

        if idx >= len(self.Gd):
            raise Exception("invalid index")

        if (self.which_mask == 'myo'):
            sample = (self.Gd[idx], self.myo_masks[idx], self.names[idx])

        if (self.which_mask == 'endo'):
            sample = (self.Gd[idx], self.endo_masks[idx], self.names[idx])

        if (self.which_mask == 'epi'):
            sample = (self.Gd[idx], self.epi_masks[idx], self.names[idx])

        if (self.which_mask == 'endo_epi'):
            sample = (self.Gd[idx], self.endo_epi_masks[idx], self.names[idx])

        if (self.which_mask == 'endo_epi_rvi'):
            sample = (
                self.Gd[idx], self.endo_epi_rvi_masks[idx], self.names[idx])

        if (self.which_mask == 'endo_epi_rv'):
            sample = (
                self.Gd[idx], self.endo_epi_rv_masks[idx], self.names[idx])

        if (self.which_mask == 'endo_epi_rv_rvi'):
            sample = (
                self.Gd[idx], self.endo_epi_rv_rvi_masks[idx], self.names[idx])

        if (self.which_mask == 'endo_epi_rv,rvi'):
            sample = (
                self.Gd[idx], (self.endo_epi_rv_masks[idx], self.rvi_pt[idx]),
                self.names[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_one_data(self, case_dir, loc, f_prefix):

        f_name = f_prefix + '.npy'
        data = np.load(os.path.join(case_dir, loc, f_name))

        print('Loaded ', f_name, data.shape)

        return data

    def load_from_numpy_array(self, Gd, endo_mask, epi_mask, \
        myo_mask, endo_epi_mask, endo_epi_rv_mask, \
        endo_epi_rv_rv_insertion_mask, endo_epi_rv_insertion_mask, \
        rv_insertion_pt):

        Gd = np.squeeze(Gd)
        # Gd = Gd / np.max(Gd)
        Gd = np.transpose(Gd, (2, 0, 1))

        endo = endo_mask
        endo = np.reshape(endo, (1, endo.shape[0], endo.shape[1]))

        epi = epi_mask
        epi = np.reshape(epi, (1, epi.shape[0], epi.shape[1]))

        myo = myo_mask
        myo = np.reshape(myo, (1, myo.shape[0], myo.shape[1]))

        endo_epi = endo_epi_mask
        endo_epi = np.reshape(endo_epi,
                              (1, endo_epi.shape[0], endo_epi.shape[1]))

        endo_epi_rv = endo_epi_rv_mask
        endo_epi_rv = np.reshape(endo_epi_rv, (
            1, endo_epi_rv.shape[0], endo_epi_rv.shape[1]))

        endo_epi_rv_rvi = endo_epi_rv_rv_insertion_mask
        endo_epi_rv_rvi = np.reshape(endo_epi_rv_rvi, (
            1, endo_epi_rv_rvi.shape[0], endo_epi_rv_rvi.shape[1]))

        endo_epi_rvi = endo_epi_rv_insertion_mask
        endo_epi_rvi = np.reshape(endo_epi_rvi, (
            1, endo_epi_rvi.shape[0], endo_epi_rvi.shape[1]))

        rvi_pt = rv_insertion_pt

        return Gd, endo, epi, myo, endo_epi, endo_epi_rv, endo_epi_rv_rvi, endo_epi_rvi, rvi_pt

    def load_from_Seg(self, Seg, ind, is_seg_norm):

        if (is_seg_norm):
            endo = Seg[0][ind]['endo_resized_mask_norm']
            epi = Seg[0][ind]['epi_resized_mask_norm']
            myo = Seg[0][ind]['myo_resized_mask_norm']
            endo_epi = Seg[0][ind]['endo_epi_resized_mask_norm']
            endo_epi_rv = Seg[0][ind]['endo_epi_rv_resized_mask_norm']
            endo_epi_rv_rvi = Seg[0][ind]['endo_epi_rv_rvi_resized_mask_norm']
            endo_epi_rvi = Seg[0][ind]['endo_epi_rvi_resized_mask_norm']
            rvi_pt = Seg[0][ind]['rvi_resized_norm']
            roi = Seg[0][ind]['roi_norm']
        else:
            endo = Seg[0][ind]['endo_resized_mask']
            epi = Seg[0][ind]['epi_resized_mask']
            myo = Seg[0][ind]['myo_resized_mask']
            endo_epi = Seg[0][ind]['endo_epi_resized_mask']
            endo_epi_rv = Seg[0][ind]['endo_epi_rv_resized_mask']
            endo_epi_rv_rvi = Seg[0][ind]['endo_epi_rv_rvi_resized_mask']
            endo_epi_rvi = Seg[0][ind]['endo_epi_rvi_resized_mask']
            rvi_pt = Seg[0][ind]['rvi_resized']
            roi = Seg[0][ind]['roi']

        '''print(Gd.shape)
        print(endo.shape)
        print(epi.shape)
        print(myo.shape)
        '''

        # Gd = Gd / np.max(Gd)

        # Gd = np.transpose(Gd, (2, 0, 1))
        '''
        endo = np.reshape(endo, (1, endo.shape[0], endo.shape[1]))
        epi = np.reshape(epi, (1, epi.shape[0], epi.shape[1]))
        myo = np.reshape(myo, (1, myo.shape[0], myo.shape[1]))
        endo_epi = np.reshape(endo_epi, (1, endo_epi.shape[0], endo_epi.shape[1]))
        endo_epi_rv = np.reshape(endo_epi_rv, (1, endo_epi_rv.shape[0], endo_epi_rv.shape[1]))
        endo_epi_rv_rvi = np.reshape(endo_epi_rv_rvi, (1, endo_epi_rv_rvi.shape[0], endo_epi_rv_rvi.shape[1]))
        endo_epi_rvi = np.reshape(endo_epi_rvi, (1, endo_epi_rvi.shape[0], endo_epi_rvi.shape[1]))
        '''

        return (
            endo, epi, myo, endo_epi, endo_epi_rv, endo_epi_rv_rvi,
            endo_epi_rvi,
            rvi_pt, roi)

    def __str__(self):
        str = "Perfusion Dataset\n"
        str += "  image root: %s" % self.img_dir + "\n"
        str += "  Number of samples: %d" % len(self.Gd) + "\n"
        str += "  Number of masks: %d" % len(self.myo_masks) + "\n"
        if len(self.Gd) > 0:
            str += "  image shape: %d %d %d" % self.Gd[0].shape + "\n"
            str += "  myo mask shape: %d %d %d" % self.myo_masks[0].shape + "\n"
            str += "  endo mask shape: %d %d %d" % self.endo_masks[
                0].shape + "\n"
            str += "  epi mask shape: %d %d %d" % self.epi_masks[0].shape + "\n"
            str += "  endo_epi mask shape: %d %d %d" % self.endo_epi_masks[
                0].shape + "\n"
            str += "  endo_epi_rv mask shape: %d %d %d" % \
                   self.endo_epi_rv_masks[0].shape + "\n"
            str += "  endo_epi_rvi mask shape: %d %d %d" % \
                   self.endo_epi_rvi_masks[0].shape + "\n"
            str += "  endo_epi_rv_rvi mask shape: %d %d %d" % \
                   self.endo_epi_rv_rvi_masks[0].shape + "\n"
            str += "  rvi_pt shape: %d %d" % self.rvi_pt[0].shape + "\n"
        return str


# %%

num_of_random = 16

# perf_dataset = PerfDatasetRandomCrop(img_dir, num_of_random=num_of_random)
print("Done")

# %%

# print(perf_dataset)

# %%

transform = torchvision.transforms.Compose(
    [RandomFlip1stDim(0.5), RandomFlip2ndDim(0.5), RandomPermute2DT(0.5)])

# %%

# perf_dataset.transform = transform

# %%

# print(perf_dataset)

# %% md

## Visual a data

# %%

# perf_dataset.which_mask = 'myo'
# sample = perf_dataset[1]
#
# print(sample[0].shape)
# print(sample[1].shape)
# print(sample[2])
#
# im = np.transpose(sample[0], [1, 2, 0])
# utils.cmr_ml_utils_plotting.plot_image_array(im[:, :, 24], columns=1,
#                                              figsize=[4, 4])
#
# a = torch.from_numpy(sample[0])
# b = torch.from_numpy(sample[1])
#
# perf_dataset.which_mask = 'endo'
# sample2 = perf_dataset[1]
#
# perf_dataset.which_mask = 'epi'
# sample3 = perf_dataset[1]
#
# perf_dataset.which_mask = 'endo_epi'
# sample4 = perf_dataset[1]
#
# perf_dataset.which_mask = 'endo_epi_rv'
# sample5 = perf_dataset[1]
#
# perf_dataset.which_mask = 'endo_epi_rv_rvi'
# sample6 = perf_dataset[1]
#
# perf_dataset.which_mask = 'endo_epi_rvi'
# sample7 = perf_dataset[1]
#
# plt.figure(figsize=(32, 32))
# plt.subplot(171)
# plt.imshow(np.squeeze(sample[1]))
# plt.subplot(172)
# plt.imshow(np.squeeze(sample2[1]))
# plt.subplot(173)
# plt.imshow(np.squeeze(sample3[1]))
# plt.subplot(174)
# plt.imshow(np.squeeze(sample4[1]))
# plt.subplot(175)
# plt.imshow(np.squeeze(sample5[1]))
# plt.subplot(176)
# plt.imshow(np.squeeze(sample6[1]))
# plt.subplot(177)
# plt.imshow(np.squeeze(sample7[1]))
#
# %% md

## Train with multi-calss segmentation

# %%

transform = torchvision.transforms.Compose(
    [RandomFlip1stDim(0.5), RandomFlip2ndDim(0.5), RandomPermute2DT(0.5)])

# %%

num_of_random = 16

# perf_dataset = PerfDatasetRandomCrop(img_dir, num_of_random=num_of_random)
print("Done")

# %%

# perf_dataset.transform = transform

# %%

# perf_dataset.transform = None

# %%

k = 12

# Chunk into k random sets
# chunks = utils.cmr_ml_utils_data.chunk(range(len(perf_dataset)), k)
# train_idxs, val_idxs = utils.cmr_ml_utils_data.get_k_fold_training_validation(
#     chunks, val_chunk=0)
#
# num_train = len(train_idxs)
# print('num_train = %d' % num_train)
# num_val = len(val_idxs)
# print('num_val = %d' % num_val)

# %%

# perf_dataset.which_mask = 'endo_epi_rv'
num_classes = 4
class_for_accu = [1, 2, 3]  # endo,epi, rv
class_weights = np.ones(num_classes)
print(class_weights)
p_thres = [0.5, 0.5, 0.5]

# %%

# print(perf_dataset.which_mask)
#
# sample = perf_dataset[1]
#
# print(sample[1].shape)
#
# plt.figure()
# plt.imshow(np.squeeze(sample[1]))

# %%

batch_size = 64

# loader_for_train = DataLoader(perf_dataset, batch_size=batch_size,
#                           sampler=sampler.SubsetRandomSampler(train_idxs))
#
# loader_for_val = DataLoader(perf_dataset, batch_size=batch_size,
#                         sampler=sampler.SubsetRandomSampler(val_idxs))
#
# iter_train = iter(loader_for_train)
#
# print(perf_dataset.which_mask)

# %%

# images, masks, names = iter_train.next()
#
# B, C, RO, E1 = images.shape
#
# print(images.shape)
# print(masks.shape)
# print(torch.max(images))
# print(torch.max(masks))
#
# plt.figure()
# plt.imshow(np.squeeze(masks[1,0,:,:]))
#
# a = images[:,0,:,:]
# print(a.shape)
# a = torch.reshape(a, (B, 1, RO, E1))
#
# plt.figure(figsize=(16, 16))
# show(make_grid(a.double(), nrow=8, padding=2, normalize=False, scale_each=True))
#
# plt.figure(figsize=(16, 16))
# show(make_grid(masks.double(), nrow=8, padding=2, normalize=True, scale_each=False))
#
# print(images.dtype)
# X = images.type(torch.FloatTensor)
# y = masks.type(torch.FloatTensor)
# print(X.shape)
# print(y.shape)
#

# %%

# from deep_learning.analysis import training
from deep_learning.analysis.training.performance import LossMulti
from deep_learning.analysis import models

# import hyper_search
# import plot_run
# from deep_learning.analysis.training import dice_coeff, centroid_diff, adaptive_thresh

num_epochs = 120
print_every = 100000

inplanes = 96
layers = [2, 3]
layers_planes = [96, 128]

# print(perf_dataset.Gd[0].shape)
C, H, W = (48, 176, 176)  # perf_dataset.Gd[0].shape

model = models.GadgetronResUnet18(F0=C,
                                  inplanes=inplanes,
                                  layers=layers,
                                  layers_planes=layers_planes,
                                  use_dropout=False,
                                  p=0.5,
                                  H=H, W=W, C=num_classes,
                                  # background, lv, myo, rv, rv insertion
                                  verbose=True)
# print(model)


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print("model on multiple GPU ... ")
    # print(model)

patience = 5
factor = 0.5
cooldown = 1
min_lr = 1e-5

weight_decay = 0
learning_rate = 1e-3

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                       eps=1e-08, weight_decay=weight_decay, amsgrad=False)

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       patience=20,
                                                       verbose=True)

criterion = LossMulti(class_weights=class_weights, jaccard_weight=0.5)
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()

log_dir = 'perf_training/ResUnet' + '_lr_' + str(
    learning_rate) + '_epochs_' + str(num_epochs)
writer = SummaryWriter(log_dir)

# %%

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# perf_trainer = training.GadgetronMultiClassSeg_Perf(model,
#                                    optimizer,
#                                    criterion,
#                                    loader_for_train,
#                                    loader_for_val,
#                                    class_for_accu=class_for_accu,
#                                    p_thres = p_thres,
#                                    scheduler=scheduler,
#                                    epochs=num_epochs,
#                                    device=device,
#                                    x_dtype=torch.float32,
#                                    y_dtype=torch.long,
#                                    early_stopping_thres = 100,
#                                    print_every=print_every,
#                                    small_data_mode = False,
#                                    writer=writer,
#                                    model_folder="perf_training/")

# %%

# epochs_traning, epochs_validation, best_model, loss_all, epochs_acc_class = perf_trainer.train(verbose=True, epoch_to_load=-1, save_model_epoch=True)

# %%

# acc, loss, acc_class = perf_trainer.check_validation_test_accuracy(loader_for_val, best_model)
# print(acc, loss)
# print(acc_class)

# %% md

## Saving the model

# %%

# try:
#     best_model_cpu = best_model.cpu().module
# except:
#
#     best_model_cpu = best_model.cpu()
#
# print(best_model_cpu)


# %%

# print(perf_dataset.transform)
v = torch.__version__
print(v)

# %%

from datetime import date

today = str(date.today())
print(today)

# %%

# if (perf_dataset.transform == None):
#     model_file = '/home/xueh2/mrprogs/gadgetron_CMR_ML-source/deployment/networks/perf_' + perf_dataset.which_mask + '_network_' + today + '_CMR_View' + '_Pytorch_' + v + '.pbt'
# else:
#     model_file = '/home/xueh2/mrprogs/gadgetron_CMR_ML-source/deployment/networks/perf_' + perf_dataset.which_mask + '_network_' + today + '_Pytorch_' + v + '.pbt'
# print(model_file)

# %%

## Correct save!
# import copy

# best_model_wts = copy.deepcopy(best_model_cpu.cpu().state_dict())
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.GadgetronResUnet18(F0=C,
                                  inplanes=inplanes,
                                  layers=layers,
                                  layers_planes=layers_planes,
                                  use_dropout=False,
                                  p=0.5,
                                  H=H, W=W, C=num_classes,
                                  verbose=True)
# model.to(device=device)

# %%

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = np.random.normal(size=(10, 48, 176, 176))
x_tensor = torch.Tensor(x, device=device)
print(model(x_tensor).shape)
# empty_model.load_state_dict(best_model_wts)

# print(empty_model)

# torch.save(empty_model.state_dict(), '/home/xueh2/cmr_ml/deployment/networks/abstract_network.dict')
# torch.save(empty_model, model_file)


# %%

# load test
# work with python application
# Runs on CPU so very slow

# model_loaded = torch.load(model_file)
#
# print(model_loaded)
