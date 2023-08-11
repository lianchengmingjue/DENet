import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch.utils.data as data

import torchvision.transforms as transforms
import albumentations.augmentations.transforms as albu_transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from albumentations import HorizontalFlip, RandomResizedCrop, Compose, DualTransform


class LOGODataset(data.Dataset):
    def __init__(self,train,args,data='_images',input_size = 256,limited_dataset=1,normalized_input=False, sample=[]):

        self.train = []
        self.anno = []
        self.mask = []
        self.wm = []
        self.base_folder = args.dataset_dir
        self.input_size = input_size
        self.normalized_input = normalized_input
        self.dataset = train+data

        self.istrain = False if self.dataset.find('train') == -1 else True
        self.sample = sample

        mypath = join(self.base_folder,self.dataset)
        file_names = sorted([f for f in listdir(join(mypath,'image')) if isfile(join(mypath,'image', f)) ])

        # 这段if好像没啥用
        if limited_dataset > 0:
            xtrain = sorted(list(set([ file_name.split('-')[0] for file_name in file_names ])))
            tmp = []
            for x in xtrain:
                # get the file_name by identifier
                tmp.append([y for y in file_names if x in y][0])

            file_names = tmp
        else:
            file_names = file_names

        for file_name in file_names:
            self.train.append(os.path.join(mypath,'image',file_name))
            self.mask.append(os.path.join(mypath,'mask',file_name))
            self.wm.append(os.path.join(mypath,'wm',file_name))
            self.anno.append(os.path.join(self.base_folder,'natural',file_name.split('-')[0]+'.jpg'))

        if len(self.sample) > 0 :
            self.train = [ self.train[i] for i in self.sample ] 
            self.mask = [ self.mask[i] for i in self.sample ] 
            self.anno = [ self.anno[i] for i in self.sample ] 

        self.trans = transforms.Compose([
                # transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor()
            ])

        print('total Dataset of '+self.dataset+' is : ', len(self.train))

        
        self.albu_transform = self.get_transform(additional_targets={'target':'image', 'wm':'image', 'mask':'mask' })

    # image加了水印之后的图片作为网络输入
    # target每加水印的图片作为网络输出

    def __getitem__(self, index):
        img = Image.open(self.train[index]).convert('RGB')
        mask = Image.open(self.mask[index]).convert('L')
        anno = Image.open(self.anno[index]).convert('RGB')
        wm = Image.open(self.wm[index]).convert('RGB')
        # origin_img = transforms.Compose([
        #         transforms.Resize((self.input_size,self.input_size)),
        #         transforms.ToTensor()
        #     ])(img)
        sample = {
                "image": np.array(img),
                "target": np.array(anno), 
                "mask": np.array(mask), 
                "wm": np.array(wm),
        }
        sample = self.augment_sample(sample)
        for key in sample:
            sample[key] = self.trans(sample[key])
        
        sample["img_path"]=self.train[index]
        # sample['origin_img']=origin_img
        return sample


    def __len__(self):

        return len(self.train)

    def get_transform(self, params=None, grayscale=False, convert=True, additional_targets=None):
        transform_list = []
        transform_list.append(albu_transforms.Resize(self.input_size, self.input_size))
        
        if self.istrain == True:
            transform_list.append(HorizontalFlip())
        
        return Compose(transform_list, additional_targets=additional_targets)
    # 数据增强

    def augment_sample(self, sample):
        #print(self.transform.additional_targets.keys())
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.albu_transform.additional_targets.keys()}

        aug_output = self.albu_transform(image=sample['image'], **additional_targets)

        for target_name, transformed_target in aug_output.items():
            #print(target_name,transformed_target.shape)
            sample[target_name] = transformed_target

        return sample


