import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import nailong_utils as utils  # 导入辅助模块


# 创建并设置主窗口
def create_main_window():
    win = tk.Tk()
    win.title("Nailong Recognition")
    win.geometry("800x700")
    win.configure(bg="#E6F7FF")  # 设置浅蓝色背景
    return win


# 创建并设置界面布局
def setup_ui(win, model):
    # 标题标签，字体改小
    title_label = tk.Label(win, text="Nailong Recognition", font=("Comic Sans MS", 18, "bold"), bg="#E6F7FF", fg="#333333")
    title_label.pack(pady=10)

    # 创建框架用于按钮布局
    button_frame = tk.Frame(win, bg="#E6F7FF")
    button_frame.pack(side="top", pady=15)

    # 显示图片的标签，设置初始注释文字并居中放置
    img_label = tk.Label(
        win, text="click \"Choose Image\" to choose\nclick \"Predict Image\" to predict",
        font=("Comic Sans MS", 18, "italic"), bg="#D3D3D3", fg="#555555",
        relief="ridge", borderwidth=2
    )
    img_label.pack(pady=20, ipadx=10, ipady=10)

    # 显示预测结果的标签，使用 place 布局固定位置
    result_label = tk.Label(win, text="", font=("Comic Sans MS", 14), bg="#E6F7FF", fg="#333333")
    result_label.place(relx=0.5, rely=0.9, anchor="center")  # 固定位置，水平居中，靠近底部

    # 定义卡通风格的按钮样式
    def create_cartoon_button(parent, text, command, bg, hover_bg):
        button = tk.Button(
            parent, text=text, font=("Comic Sans MS", 12), bg=bg, fg="white",
            activebackground=hover_bg, relief="groove", borderwidth=4,
            command=command
        )
        return button

    # 按钮用于选择图像
    img_button = create_cartoon_button(
        button_frame, "Choose Image", lambda: load_image(img_label),
        bg="#FFA07A", hover_bg="#FF6347"
    )
    img_button.pack(side="left", padx=10)

    # 按钮用于进行预测
    predict_button = create_cartoon_button(
        button_frame, "Predict Image", lambda: predict_image(img_label, result_label, model),
        bg="#87CEFA", hover_bg="#4682B4"
    )
    predict_button.pack(side="left", padx=10)

    # 右下角添加注释文本
    footer_label = tk.Label(win, text="Made by Zxuuuu just for fun", font=("Arial", 10), bg="#E6F7FF", fg="#666666")
    footer_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)  # 设置在窗口右下角


# 选择图片并显示
# 选择图片并显示
def load_image(img_label):
    global current_image_path
    file_path = filedialog.askopenfilename()
    if file_path:
        # 加载并显示图像
        image = Image.open(file_path).convert("RGB")
        image = image.resize((400, 400), Image.Resampling.LANCZOS)  # 确保图片尺寸为 400x400
        image_tk = ImageTk.PhotoImage(image)
        img_label.config(image=image_tk, text="")  # 移除默认文本并显示图片
        img_label.image = image_tk
        current_image_path = file_path  # 保存路径供预测使用

        # 恢复原始背景颜色
        img_label.master.configure(bg="#E6F7FF")  # 设置为浅蓝色背景


# 进行预测并显示结果
# 进行预测并显示结果
def predict_image(img_label, result_label, model):
    if 'current_image_path' not in globals():
        result_label.config(text="Please select an image first.")
        return
    image_tensor = utils.preprocess_image(current_image_path)
    predicted_label = utils.predict_image(model, image_tensor)
    result_label.config(text=f"{utils.label_list[predicted_label]}")

    # 如果label为1，开始摇晃效果并改变背景颜色
    if predicted_label == 1:
        shake_image(img_label)  # 调用摇晃函数
        img_label.master.configure(bg="#FFDAB9")  # 更改背景为淡橙色
        enlarge_and_color_text(result_label)  # 调用字幕放大和彩色效果

    # 播放音乐
    if predicted_label == 1:
        utils.play_music("nailong.mp3")  # 播放音乐


# 实现字幕放大和变色效果
def enlarge_and_color_text(label):
    # 设置放大和彩色效果
    label.config(font=("Comic Sans MS", 24, "bold"), fg="red")  # 放大并设置为红色

    def reset_text_style():
        # 恢复原始字体样式
        label.config(font=("Comic Sans MS", 14), fg="#333333")

    # 在 2 秒后恢复原样
    label.after(2000, reset_text_style)


# 实现摇晃效果
def shake_image(img_label):
    original_x = img_label.winfo_x()  # 获取当前图片的位置
    original_y = img_label.winfo_y()  # 获取图片的竖直位置
    shake_distance = 10  # 摇晃的幅度
    shake_times = 10  # 摇晃的次数
    shake_interval = 50  # 摇晃的时间间隔（毫秒）

    def shake(step):
        if step < shake_times:
            # 计算新的x坐标，模拟摇晃
            offset = shake_distance if step % 2 == 0 else -shake_distance
            img_label.place(x=original_x + offset, y=original_y)  # 仅改变x坐标，保持y坐标不变
            img_label.after(shake_interval, shake, step + 1)  # 定时调用下一次移动
        else:
            img_label.place(x=original_x, y=original_y)  # 摇晃完成后恢复到原位置

    shake(0)  # 开始摇晃


# 主程序入口
if __name__ == "__main__":
    win = create_main_window()
    model = utils.load_model()
    setup_ui(win, model)
    win.mainloop()
