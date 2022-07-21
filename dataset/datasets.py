from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
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
        print("image_names = ", self.image_names)

        self.label_names, self.flow_names = io.get_label_files(self.image_names, self.mask_filter, imf=self.imf)
        print("label_names = ", self.label_names)
        print("flow_names = ", self.flow_names)

    def __getitem__(self, i):
        image = io.imread(self.image_names[i])
        image = np.asarray(image[None,:,:])[self.channels] if image.ndim==2 else np.asarray(image.transpose(2,0,1))[self.channels]
        label = io.imread(self.label_names[i])


        if self.flow_names:
            print("************ flows precomputed *************")
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

        print("image.shape = ", image.shape)
        print("flow.shape = ", flow.shape)

        return image.astype(np.float32)/255, flow

    def __len__(self):
        return len(self.image_names)

# 运行方式：进入上一层文件夹，然后以模块方式运行，即：python -m dataset.datasets
if __name__ == '__main__':
    dataset = CellposeDataset(data_dir="./data/", chan=0, chan2=0, img_filter='_img', mask_filter='_mask')
    print(len(dataset))