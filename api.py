import nailong_UI
import nailong_utils as utils

if __name__ == "__main__":
    # 初始化主窗口和模型
    win = nailong_UI.create_main_window()
    model = utils.load_model()

    nailong_UI.setup_ui(win, model)
    win.mainloop()
