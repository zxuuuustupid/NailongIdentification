import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SimpleImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 包含所有类别子文件夹的根目录
        :param transform: 应用于每个样本的可选变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        label_code=0
        # 遍历每个类别的文件夹
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            for image_name in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, image_name))
                self.labels.append(label_code)  # 假设文件夹名即为标签
            label_code=label_code+1

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        根据给定的索引返回一个样本
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")  # 转换为RGB
        image = self.transform(image)
        label = self.labels[index]

        return image, label



