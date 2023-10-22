import os
import numpy as np
import torch
import torchvision
from PIL import Image
import seaborn as sns

class MNIST_LT(torchvision.datasets.MNIST):
    cls_num = 10
    
    def __init__(self, root, imb_type='exp', imb_factor=0.01, noise_mode='imb', noise_ratio=0.25, resample_ratio=1.0,
                 rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_LT, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)

        if train:
            self.resample_ratio = resample_ratio
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.get_noisy_data(self.cls_num, noise_mode, noise_ratio)
        
        self.labels = self.targets
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num  * self.resample_ratio
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        
        _, cur_img_nums = np.unique(self.targets, return_counts=True)
        for cls_idx in range(cls_num):
            img_num_per_cls[cls_idx] = min(img_num_per_cls[cls_idx], cur_img_nums[cls_idx])

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = torch.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
    
    def get_noisy_data(self, cls_num, noise_mode, noise_ratio):
        train_label = self.targets
        
        noise_label = []
        num_train = len(self.targets)
        idx = list(range(num_train))
        np.random.shuffle(idx)
        cls_num_list = self.get_cls_num_list()

        if noise_mode == 'imb':
            num_noise = int(noise_ratio * num_train)
            noise_idx = idx[:num_noise]

            p = np.array([cls_num_list for _ in range(cls_num)])
            for i in range(cls_num):
                p[i][i] = 0
            p = p / p.sum(axis=1, keepdims=True)
            for i in range(num_train):
                if i in noise_idx:
                    newlabel = np.random.choice(cls_num, p=p[train_label[i]])
                    assert newlabel != train_label[i]
                    noise_label.append(newlabel)
                else:    
                    noise_label.append(train_label[i])

        noise_label = np.array(noise_label, dtype=np.int8).tolist()

        self.clean_targets = self.targets[:]
        self.targets = noise_label

        for c1, c0 in zip(self.targets, self.clean_targets):
            if c1 != c0:
                self.num_per_cls_dict[c1] += 1
                self.num_per_cls_dict[c0] -= 1

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')


class FashionMNIST_LT(torchvision.datasets.FashionMNIST, MNIST_LT):
    cls_num = 10

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'FashionMNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'FashionMNIST', 'processed')


class CMNIST_LT(MNIST_LT):
    cls_num = 10
    
    def __init__(self, root, imb_type='exp', imb_factor=0.01, noise_mode='imb', noise_ratio=0.25, resample_ratio=1.0,
                 rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(CMNIST_LT, self).__init__(root,
            imb_type=imb_type, imb_factor=imb_factor, noise_mode=noise_mode, noise_ratio=noise_ratio, resample_ratio=resample_ratio,
            rand_number=rand_number, train=train, transform=transform, target_transform=target_transform, download=download)

        colors = sns.color_palette() # [10, 3]
        self.colors = np.array(colors)

        if train:
            p = np.ones([self.cls_num, self.cls_num]) * (1.0 / self.cls_num)
            for k in range(5, 10):
                p[k] *= 0
                p[k][k] = 1.0

            self.data = torch.stack([self.data] * 3, dim=-1)
            self.get_colored_data(self.cls_num, p)
        else:
            p = np.ones([self.cls_num, self.cls_num]) * (1.0 / self.cls_num)
            
            self.data = torch.stack([self.data] * 3, dim=-1)
            self.get_colored_data(self.cls_num, p)
    
    def get_colored_data(self, cls_num, trans_mat):
        num_train = len(self.targets)
        for i in range(num_train):
            color_index = np.random.choice(cls_num, p=trans_mat[self.targets[i]])
            self.data[i] = self.data[i] * self.colors[color_index]
            
    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target