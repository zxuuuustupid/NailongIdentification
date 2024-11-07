import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms
from nailong.model.CNN import VGG

label_list=['不是奶龙', "是啥比奶龙"]
# 初始化模型
model = VGG()
model.load_state_dict(torch.load("model/model_weights.pth"))
model.eval()  # 切换到评估模式

# 创建窗口
win = tk.Tk()
win.title("Image Recognition")
win.geometry("800x600")

# 创建一个框架，将按钮放置在顶部并保持水平排列
button_frame = tk.Frame(win)
button_frame.pack(side="top", pady=20)

# 显示图片的标签
img_label = tk.Label(win)
img_label.pack(pady=20)

# 显示预测结果的标签
result_label = tk.Label(win, text="")
result_label.pack(pady=20)

# 预处理步骤：将图像转换为Tensor并进行标准化
preprocess = transforms.Compose([
    transforms.Resize((224,224)),  # 调整图像大小为224x224，这是VGG模型的标准输入尺寸
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
])


# 按钮用于选择图片
def load_image():
    global current_image_path  # 声明当前图片路径为全局变量
    file_path = filedialog.askopenfilename()
    if file_path:
        # 加载图像并确保是 RGB 模式
        image = Image.open(file_path).convert("RGB")  # 确保是RGB格式
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)

        # 更新标签显示图像
        img_label.config(image=image)
        img_label.image = image

        # 保存文件路径，供后续预测使用
        current_image_path = file_path

# 按钮用于进行预测
def predict_image():
    if 'current_image_path' not in globals():
        result_label.config(text="Please select an image first.")
        return

    file_path = current_image_path
    if file_path:
        image = Image.open(file_path).convert("RGB")  # 确保是RGB格式
        image = preprocess(image).unsqueeze(0)  # 转换为Tensor并增加批次维度
        print(image.shape)
        # 检查输入的图像维度，确保其形状是 (1, 3, 224, 224)

        if image.shape[1] != 3 or image.shape[2] != 224 or image.shape[3] != 224:
            result_label.config(text="Error: The image must have size (224, 224) and 3 channels (RGB).")
            return

        # 将图像传入模型进行预测
        with torch.no_grad():  # 禁用梯度计算
            output = model(image)  # 通过模型进行推理

        # 获取预测结果（假设是分类问题）
        _, predicted_idx = torch.max(output, 1)

        # 显示预测结果
        predicted_label = predicted_idx.item()  # 获取预测类别的索引
        result_label.config(text=f"Prediction: {label_list[predicted_label]}")

# 按钮选择图像
img_button = tk.Button(button_frame, text="Choose Image", command=load_image)
img_button.pack(side="left", padx=10)

# 按钮进行预测
predict_button = tk.Button(button_frame, text="Predict Image", command=predict_image)
predict_button.pack(side="left", padx=10)

# 运行主循环
win.mainloop()
