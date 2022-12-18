from torch.utils.data import Dataset
import h5py
from PIL import Image
import os
import torchvision.transforms as transforms
import torch

import random


def cal_boundary(mean, std):
    """
    calculate the maximum and minimum in each channel after the normalziation
    :param mean: the mean values for RGB channels
    :param std: the std values for RGB channels
    :return: max_in, min_in are maximum and minimum in corresponding channel
    """
    max_image, min_image = 1, 0
    max_in, min_in = [], []
    for i in range(3):
        max_in.append((max_image - mean[i]) / std[i])
        min_in.append((min_image - mean[i]) / std[i])

    return max_in, min_in


class Processing_pk():
    def __init__(self, mean, std, frame_shape):
        self.mean = mean
        self.std = std
        self.frame_shape = frame_shape
        self.max_in, self.min_in = cal_boundary(mean, std)
        self.normalize = transforms.Normalize(mean, std)
        self.inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )

    def mytransform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.frame_shape, Image.BICUBIC),
                transforms.ToTensor()
            ]
        )

    def mypreprocess(self):
        return self.normalize

    def myinvtransform(self):
        return transforms.ToPILImage()

    def myinvpreprocessing(self):
        return self.inv_normalize

def get_params(input_shape, output_size):
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = input_shape
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

class VideoDataset_tra(Dataset):
    def __init__(self, root, clips_index, seq_len, target_map, preprocess, DA_flag=None):
        self.root = root
        self.keys, self.records, self.att_size, self.real_size = self._readClips_index(clips_index)
        # check seq_len == len(records[0])-1. If no, re-generate the clips_index by datasetClips
        assert seq_len == len(
            self.records[0]) - 1, f'the length of clips is not {seq_len}, re-generate the clips_index by datasetClips'
        self.seq_len = seq_len
        self.target_map = target_map
        self.preprocess = preprocess
        self.phrase = self._parse_indexFile(clips_index.split('/')[-1].split('\\')[-1])
        self.DA_flag = DA_flag

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        key = self.keys[index]
        frames_PIL = []
        for i in range(self.seq_len):
            if self.phrase is None:
                path = os.path.join(self.root, sample[i].decode("ascii", "ignore")).replace('\\', '/')
            else:
                path = os.path.join(self.root, self.phrase, sample[i].decode("ascii", "ignore")).replace('\\', '/')
            frames_PIL.append(Image.open(path))

        target = self.target_map[sample[-1].decode("ascii", "ignore")]


        if self.DA_flag:
            video = self._video_transform_DA(frames_PIL)
        else:
            video = self._video_transform_without_DA(frames_PIL)

        if self.preprocess is not None:
            video_ = []
            for i in range(self.seq_len):
                video_.append(self.preprocess(video[i]))
            video_ = torch.stack(video_)
        else:
            video_ = video

        return video_, target, key

    def _video_transform_DA(self, frames_PIL):
        """
        Resize(resize_shape) -> (250*250) -> RandomCrop(crop_size) -> (224*224) -> RandomHorizontalFlip -> ToTensor
        """
        resize_shape = 280
        crop_size = 224
        i, j, th, tw = get_params((resize_shape, resize_shape), (crop_size, crop_size))
        myrandomcrop = transforms.MyRandomCrop(crop_size, i, j, th, tw)
        flip_rate = 0.5
        flip_flag = torch.rand(1) < flip_rate
        myrandomhorizontalflip = transforms.MyRandomHorizontalFlip(flip_flag)

        video_transforms = transforms.Compose(
            [
                transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
                myrandomcrop,
                myrandomhorizontalflip,
                transforms.ToTensor()
            ]
        )

        frames_tensor = []
        for i in range(len(frames_PIL)):
            frame = frames_PIL[i]
            frames_tensor.append(video_transforms(frame))
        frames_tensor = torch.stack(frames_tensor)

        return frames_tensor

    def _video_transform_without_DA(self, frames_PIL):
        """
        Resize(resize_shape) -> (250*250) -> RandomCrop(crop_size) -> (224*224) -> RandomHorizontalFlip -> ToTensor
        """
        resize_shape = 224
        # crop_size = 224
        # i, j, th, tw = get_params((resize_shape, resize_shape), (crop_size, crop_size))
        # myrandomcrop = transforms.MyRandomCrop(crop_size, i, j, th, tw)
        # flip_rate = 0.5
        # flip_flag = torch.rand(1) < flip_rate
        # myrandomhorizontalflip = transforms.MyRandomHorizontalFlip(flip_flag)

        video_transforms = transforms.Compose(
            [
                transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
                # myrandomcrop,
                # myrandomhorizontalflip,
                transforms.ToTensor()
            ]
        )

        frames_tensor = []
        for i in range(len(frames_PIL)):
            frame = frames_PIL[i]
            frames_tensor.append(video_transforms(frame))
        frames_tensor = torch.stack(frames_tensor)

        return frames_tensor

    def _readClips_index(self, clips_index):
        """
        clips_index is the file output from datasetClips
        Three outputs of this function
        records: a list of samples in the clips_index, including [frame_per_clip+1] strings
        att_size: the number of attack samples in the clips_index
        real_size: the number of real samples in the clips_index
        """
        records = []
        f = h5py.File(clips_index, 'r')
        att_size = f.attrs['att_size']
        real_size = f.attrs['real_size']
        keys = list(f.keys())
        for key in keys:
            records += list(f[key])

        return keys, records, att_size, real_size

    def _parse_indexFile(self, clips_index):
        if clips_index.split("_")[1].split(".")[0] == 'tra':
            return 'train'
        else:
            return clips_index.split("_")[1].split(".")[0]


class VideoDataset_test(Dataset):
    def __init__(self, root, clips_index, seq_len, target_map, transforms, preprocess):
        self.root = root
        self.keys, self.records, self.att_size, self.real_size = self._readClips_index(clips_index)
        # check seq_len == len(records[0])-1. If no, re-generate the clips_index by datasetClips
        assert seq_len == len(
            self.records[0]) - 1, f'the length of clips is not {seq_len}, re-generate the clips_index by datasetClips'
        self.seq_len = seq_len
        self.target_map = target_map
        self.transforms = transforms
        self.preprocess = preprocess
        self.phrase = self._parse_indexFile(clips_index.split('/')[-1].split('\\')[-1])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        key = self.keys[index]
        frames_PIL = []
        for i in range(self.seq_len):
            if self.phrase is None:
                path = os.path.join(self.root, sample[i].decode("ascii", "ignore")).replace('\\', '/')
            else:
                path = os.path.join(self.root, self.phrase, sample[i].decode("ascii", "ignore")).replace('\\', '/')
            frames_PIL.append(self.transforms(Image.open(path)))

        video = torch.stack(frames_PIL)
        target = self.target_map[sample[-1].decode("ascii", "ignore")]

        if self.preprocess is not None:
            video_ = []
            for i in range(self.seq_len):
                video_.append(self.preprocess(video[i]))
            video_ = torch.stack(video_)
        else:
            video_ = video

        return video_, target, key


    def _readClips_index(self, clips_index):
        """
        clips_index is the file output from datasetClips
        Three outputs of this function
        records: a list of samples in the clips_index, including [frame_per_clip+1] strings
        att_size: the number of attack samples in the clips_index
        real_size: the number of real samples in the clips_index
        """
        records = []
        f = h5py.File(clips_index, 'r')
        att_size = f.attrs['att_size']
        real_size = f.attrs['real_size']
        keys = list(f.keys())
        for key in keys:
            records += list(f[key])

        return keys, records, att_size, real_size

    def _parse_indexFile(self, clips_index):
        if clips_index.split("_")[0] == 'msu':
            return None
        if clips_index.split("_")[1].split(".")[0] == 'tra':
            return 'train'
        elif clips_index.split("_")[1].split(".")[0] == 'traAE':
            return 'train_perturb_E_01_iter_50'
        else:
            return clips_index.split("_")[1].split(".")[0]


