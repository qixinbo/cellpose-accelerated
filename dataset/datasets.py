from accelerate.state import torch
from torch.utils.data import Dataset, random_split
import albumentations as A

import cv2
import numpy as np
import os
import tifffile

from . import io, dynamics

transform = A.Compose([
    A.Resize(height=224, width=224, interpolation=cv2.INTER_NEAREST, p=1)
    ])

class CellposeDataset(Dataset):
    def __init__(self, data_dir, chan=0, chan2=0, img_filter='_img', mask_filter='_mask', look_one_level_down=False):
        self.channels = [chan, chan2]
        self.imf = img_filter
        self.mask_filter = mask_filter

        self.transform = transform
        self.data_dir = data_dir

        self.image_names = []
        self.label_names = []
        self.flow_names = []

        self.image_names = io.get_image_files(self.data_dir, self.mask_filter, imf=self.imf, look_one_level_down=look_one_level_down)

        self.label_names, self.flow_names = io.get_label_files(self.image_names, self.mask_filter, imf=self.imf)

    def __getitem__(self, i):
        image = cv2.imread(self.image_names[i], cv2.IMREAD_COLOR)
        image = np.asarray(image[None,:,:])[self.channels] if image.ndim==2 else np.asarray(image.transpose(2,0,1))[self.channels]
        label = io.imread(self.label_names[i])

        print("self.flow_names = ", self.flow_names)

        if self.flow_names:
            flow = io.imread(self.flow_names[i])
            if flow.shape[0]<4:
                flow = np.concatenate((label[np.newaxis,:,:], flow), axis=0) 
        else:
            # #!!!!!!!!!!!!!!!!!Don't forget to delete generated files
            print("------------ computing flows for labels ----------")
            veci = dynamics.masks_to_flows(label, use_gpu=False)
            flow = np.concatenate((label[None, :, :], label[None, :, :]>0.5, veci), axis=0).astype(np.float32)
            file_name = os.path.splitext(self.image_names[i])[0]
            tifffile.imsave(file_name+'_flows.tif', flow)

        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0), mask=flow.transpose(1, 2, 0))
            image = transformed["image"].transpose(2, 0, 1)
            flow = transformed["mask"].transpose(2, 0, 1)

        return image.astype(np.float32)/255, flow

    def __len__(self):
        return len(self.image_names)


def createCellposeDataset(data_dir, validation_split=None, chan=1, chan2=0, img_filter='_img', mask_filter='_mask', look_one_level_down=False):
    dataset = CellposeDataset(data_dir, chan, chan2, img_filter, mask_filter, look_one_level_down)
    if validation_split is not None:
        num_trainset = int((1-validation_split)*len(dataset))
        num_validset = len(dataset) - num_trainset
        return random_split(dataset, [num_trainset, num_validset])
    else:
        return dataset


# 运行方式：进入上一层文件夹，然后以模块方式运行，即：python -m dataset.datasets
if __name__ == '__main__':
    dataset = CellposeDataset(data_dir="./data/test", chan=1, chan2=0, img_filter='_img', mask_filter='_masks')
    print(len(dataset))

    # this is necessary for calculating all the flow files
    for i in range(len(dataset)):
        _, _ = dataset[i]

    import matplotlib.pyplot as plt
    img, flow = dataset[0]
    print("img.shape = ", img.shape)
    print("img[0] = ", img[0])
    print("max of img[0] = ", np.max(img[0]))
    print("min of img[0] = ", np.min(img[0]))

    print("flow.shape = ", flow.shape)
    print("flow[2] = ", flow[2])
    print("max of flow[2] = ", np.max(flow[2]))
    print("min of flow[2] = ", np.min(flow[2]))
    print("flow[3] = ", flow[3])
    print("max of flow[3] = ", np.max(flow[3]))
    print("min of flow[3] = ", np.min(flow[3]))

    img0 = img[0][:, :, np.newaxis]*255
    plt.imshow(img0)
    # plt.show()
