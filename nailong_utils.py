# nailong_utils.py

import torch
from PIL import Image
from torchvision import transforms
import pygame
from nailong.model.CNN import VGG

label_list = ["Relax, normal image", "Nailong!!! what the fuck"]

# 初始化 Pygame
pygame.init()
pygame.mixer.init()

# 初始化模型
def load_model():
    model = VGG()
    model.load_state_dict(torch.load("model/model_weights.pth"))
    model.eval()  # 切换到评估模式
    return model

# 预处理图像
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# 进行预测
def predict_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

# 播放音乐
def play_music(music_file):
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()
