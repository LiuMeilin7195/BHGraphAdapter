import os
import json
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

class ImageNetSketch(VisionDataset):
    def __init__(self, root, transform=None):
        super(ImageNetSketch, self).__init__(root, transform=transform)
        
        # 读取类别名称到synset ID的映射
        self.classnames = {}
        with open(os.path.join(root, 'classnames.txt'), 'r') as f:
            for line in f:
                synset, classname = line.strip().split(' ', 1)
                self.classnames[synset] = classname
        
        # 读取类别名称到标签的映射
        with open(os.path.join(root, 'classname_label.json'), 'r') as f:
            self.class_to_idx = json.load(f)
        
        # 构建synset ID到标签的映射
        self.synset_to_idx = {}
        for synset, classname in self.classnames.items():
            self.synset_to_idx[synset] = self.class_to_idx[classname]
        
        # 构建数据集
        self.samples = []
        self.targets = []
        self.imgs = []  # 与ImageNet格式保持一致
        image_dir = os.path.join(root, 'images')
        
        # 遍历所有类别文件夹
        for synset in self.synset_to_idx.keys():
            class_dir = os.path.join(image_dir, synset)
            if os.path.exists(class_dir):
                class_idx = self.synset_to_idx[synset]
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
                        self.imgs.append((img_path, class_idx))
                        self.targets.append(class_idx)
        
        self.targets = torch.LongTensor(self.targets)
        
        print(f"Loaded ImageNet-Sketch dataset with {len(self.samples)} images")
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, target
    
    def __len__(self):
        return len(self.samples)