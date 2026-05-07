import os
import torch
from torchvision.datasets import VisionDataset
from PIL import Image

class ImageNetV2(VisionDataset):
    def __init__(self, root, transform=None):
        super(ImageNetV2, self).__init__(root, transform=transform)
        
        # 读取类别映射文件
        self.class_map = {}
        with open(os.path.join(root, 'classnames.txt'), 'r') as f:
            for idx, line in enumerate(f):
                synset = line.strip().split()[0]
                self.class_map[str(idx)] = synset
        
        # 构建数据集
        self.samples = []
        self.targets = []
        base_path = os.path.join(root, 'imagenetv2-matched-frequency-format-val')
        
        for class_idx in range(1000):  # ImageNetV2有1000个类别
            class_dir = os.path.join(base_path, str(class_idx))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
                        self.targets.append(class_idx)
                        
        self.targets = torch.LongTensor(self.targets)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, target
    
    def __len__(self):
        return len(self.samples)