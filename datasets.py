import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import random


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None, need_random = False):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']
        self.need_random = need_random

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load image text information (completely into memory)
        with open(os.path.join(data_folder, self.split + '_IMAGE_TEXT_' + data_name + '.json'), 'r') as j:
            self.image_texts = json.load(j)


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        if transform == "svg":
            self.transform = None
            self.image_type = "svg"
        else:
            self.image_type = "pixel"

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        self.order = [i for i in range(14)] 

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        if self.image_type == "svg":
            img_numpy = self.imgs[i // self.cpi]
            if self.need_random:
                print(img_numpy.shape)
                random.shuffle(self.order)
                print(self.order)
                img_numpy = img_numpy[self.order, :]
            img = torch.FloatTensor(img_numpy)
        else:
            img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
            
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        image_text = torch.LongTensor(self.image_texts[i // self.cpi])

        caplen = torch.LongTensor([self.caplens[i]])

        # 此处有修改，增加了输入的数据格式，除了图像和相应的caption之外，图像中的文字也得到了相应的输入 

        if self.split is 'TRAIN':
            return img, caption, caplen, image_text
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, image_text, all_captions

    def __len__(self):
        return self.dataset_size
