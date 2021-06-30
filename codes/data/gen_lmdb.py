#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
----------------------------
@ Author: JoefZhou         -
@ Home Page: www.zhoef.com -
@ From: tencent, UESTC     -
----------------------------
@ Date: 2021/6/30
@ Project Name InvDN
----------------------------
@ function:

@ Version:

"""
import os
from glob import glob
import cv2
import numpy as np
import lmdb
import argparse
import pickle
from tqdm import tqdm


def gen_train_lmdb(args):
    path_all_noisy = glob(os.path.join(args.data_dir, '**/*NOISY*.PNG'), recursive=True)
    path_all_noisy = sorted(path_all_noisy)
    path_all_gt = [x.replace('NOISY', 'GT') for x in path_all_noisy]
    print('Number of big images: {:d}'.format(len(path_all_gt)))

    print('Training: Split the original images to small ones!')
    path_h5 = os.path.join(args.data_dir, 'medium_imgs_train.hdf5')
    if os.path.exists(path_h5):
        os.remove(path_h5)
    pch_size = 512
    stride = 512-128
    num_patch = 0
    C = 3
    dataroot_Noisy = "/youtu_action_data/denoise/sidd/lmdb/Noisy/"
    if not os.path.exists(dataroot_Noisy):
        os.makedirs(dataroot_Noisy)
    env = lmdb.open(os.path.join(dataroot_Noisy, 'medium_imgs_train'), map_size=int(1099511627776))
    dict_data = {}
    keys = []
    with env.begin(write=True) as txn:
        for ii in tqdm(range(len(path_all_gt))):
            im_noisy_int8 = cv2.imread(path_all_noisy[ii])[:, :, ::-1]
            H, W, _ = im_noisy_int8.shape
            # im_gt_int8 = cv2.imread(path_all_gt[ii])[:, :, ::-1]
            ind_H = list(range(0, H - pch_size + 1, stride))
            if ind_H[-1] < H - pch_size:
                ind_H.append(H - pch_size)
            ind_W = list(range(0, W - pch_size + 1, stride))
            if ind_W[-1] < W - pch_size:
                ind_W.append(W - pch_size)
            inner_pch_num = 0
            for start_H in ind_H:
                for start_W in ind_W:
                    pch_noisy = im_noisy_int8[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                    # pch_gt = im_gt_int8[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                    # pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
                    pch_noisy = pch_noisy.tobytes()

                    key_ = path_all_noisy[ii].split(".")[0] + "_" + str(inner_pch_num)
                    keys.append(key_)

                    txn.put(key_.encode('ascii'), pch_noisy)
                    # h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                    #                        dtype=pch_imgs.dtype, data=pch_imgs)
                    num_patch += 1
                    inner_pch_num += 1

    dict_data["keys"] = keys
    dict_data["resolution"] = "3_%d_%d" % (pch_size, pch_size)
    with open(os.path.join(dataroot_Noisy, "meta_info.pkl"), 'wb') as fo:
        pickle.dump(dict_data, fo)

    print('Total {:d} small Noise images in training set'.format(num_patch))
    print('Finish!\n')

    dataroot_GT = "/youtu_action_data/denoise/sidd/lmdb/GT/"
    if not os.path.exists(dataroot_GT):
        os.makedirs(dataroot_GT)
    env = lmdb.open(os.path.join(dataroot_GT, 'medium_imgs_train'),  map_size=int(1099511627776))
    dict_data = {}
    keys = []
    num_patch = 0
    with env.begin(write=True) as txn:
        for ii in tqdm(range(len(path_all_gt))):
            # im_noisy_int8 = cv2.imread(path_all_noisy[ii])[:, :, ::-1]
            im_gt_int8 = cv2.imread(path_all_gt[ii])[:, :, ::-1]
            H, W, _ = im_gt_int8.shape
            ind_H = list(range(0, H - pch_size + 1, stride))
            if ind_H[-1] < H - pch_size:
                ind_H.append(H - pch_size)
            ind_W = list(range(0, W - pch_size + 1, stride))
            if ind_W[-1] < W - pch_size:
                ind_W.append(W - pch_size)
            inner_pch_num = 0
            for start_H in ind_H:
                for start_W in ind_W:
                    # pch_noisy = im_noisy_int8[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                    pch_gt = im_gt_int8[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                    # pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
                    pch_gt = pch_gt.tobytes()

                    key_ = path_all_gt[ii].split(".")[0] + "_" + str(inner_pch_num)
                    keys.append(key_)

                    txn.put(key_.encode('ascii'), pch_gt)
                    # h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                    #                        dtype=pch_imgs.dtype, data=pch_imgs)
                    num_patch += 1
                    inner_pch_num += 1

    dict_data["keys"] = keys
    dict_data["resolution"] = "3_%d_%d" % (pch_size, pch_size)
    with open(os.path.join(dataroot_GT, "meta_info.pkl"), 'wb') as fo:
        pickle.dump(dict_data, fo)

    print('Total {:d} small Noise images in training set'.format(num_patch))
    print('Finish!\n')


    # with h5.File(path_h5, 'w') as h5_file:


def gen_val_lmdb(args):
    from scipy.io import loadmat
    val_data_dict = loadmat(os.path.join(args.data_dir, 'ValidationNoisyBlocksSrgb.mat'))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksSrgb']
    val_data_dict = loadmat(os.path.join(args.data_dir, 'ValidationGtBlocksSrgb.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksSrgb']
    num_img, num_block, _, _, _ = val_data_gt.shape

    dataroot_N = '/youtu_action_data/denoise/sidd/lmdb/val/Noisy'
    if not os.path.exists(dataroot_N):
        os.makedirs(dataroot_N)
    print('Validation: Saving the noisy blocks to lmdb format!')
    env = lmdb.open(os.path.join(dataroot_N, 'medium_imgs_val_lmdb'), map_size=int(1099511627776))
    dict_data = {}
    keys = []
    num_patch = 0
    with env.begin(write=True) as txn:
        for ii in range(num_img):
            for jj in range(num_block):
                if (num_patch + 1) % 100 == 0:
                    print('    The {:d} images'.format(num_patch + 1))
                im_noisy = val_data_noisy[ii, jj,]
                im_noisy = im_noisy.tobytes()
                key_ = str(num_patch)
                keys.append(key_)
                txn.put(key_.encode('ascii'), im_noisy)
                num_patch += 1
    dict_data["keys"] = keys
    dict_data["resolution"] = "3_%d_%d" % (256, 256)
    with open(os.path.join(dataroot_N, 'medium_imgs_val_lmdb', "meta_info.pkl"), 'wb') as fo:
        pickle.dump(dict_data, fo)
    print('Finish!\n')

    dataroot_GT = '/youtu_action_data/denoise/sidd/lmdb/val/GT'
    if not os.path.exists(dataroot_GT):
        os.makedirs(dataroot_GT)
    env = lmdb.open(os.path.join(dataroot_GT, 'medium_imgs_val_lmdb'), map_size=int(1099511627776))
    dict_data = {}
    keys = []
    num_patch = 0
    with env.begin(write=True) as txn:
        for ii in range(num_img):
            for jj in range(num_block):
                if (num_patch + 1) % 100 == 0:
                    print('    The {:d} images'.format(num_patch + 1))
                im_gt = val_data_gt[ii, jj,]
                im_gt = im_gt.tobytes()
                key_ = str(num_patch)
                keys.append(key_)
                txn.put(key_.encode('ascii'), im_gt)
                num_patch += 1
    dict_data["keys"] = keys
    dict_data["resolution"] = "3_%d_%d" % (256, 256)
    with open(os.path.join(dataroot_GT, 'medium_imgs_val_lmdb', "meta_info.pkl"), 'wb') as fo:
        pickle.dump(dict_data, fo)
    print('Finish!\n')

if __name__ == '__main__':
    # /youtu_action_data/denoise/sidd/SIDD_Medium_Srgb/Data
    parser = argparse.ArgumentParser(prog='SIDD Train dataset Generation')
    # The orignal SIDD images: /ssd1t/SIDD/
    parser.add_argument('--data_dir', default="/youtu_action_data/denoise/sidd", type=str,
                        metavar='PATH',
                        help="path to save the training set of SIDD, (default: None)")
    args = parser.parse_args()
    # gen_train_lmdb(args)
    gen_val_lmdb(args)