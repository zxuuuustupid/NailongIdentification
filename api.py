import nailong_UI
import nailong_utils as utils
# import sys
# import os
#
# sys.path.append('D:\Projects\\torch\\nailong')

if __name__ == "__main__":
    win = nailong_UI.create_main_window()
    model = utils.load_model()

    nailong_UI.setup_ui(win, model)
    win.mainloop()
