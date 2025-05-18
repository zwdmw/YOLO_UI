import sys
import traceback
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow
from utils.splash_screen import showSplashScreen
from utils.theme_manager import ThemeManager
from utils.error_manager import ErrorManager
from utils.shortcut_manager import ShortcutManager
from utils.tooltip_manager import TooltipManager

# 在您的主程序开头添加（在导入 ultralytics 之前）
os.environ["USE_FLASH_ATTN"] = "0"

def main():
    try:
        # 安装全局异常处理程序
        ErrorManager.install_global_handler()
        
        # 确保数据目录存在
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 创建应用程序
        app = QApplication(sys.argv)
        
        # 创建主窗口（但尚未显示）
        window = MainWindow()
        
        # 显示启动画面
        splash = showSplashScreen(app, window)
        
   
        
        # 设置快捷键
        shortcut_manager = ShortcutManager(window)
        
        # 应用工具提示
        TooltipManager.apply_tooltips(window)
        
        # 打印欢迎消息（会在终端重定向到UI后显示）
        print("YOLO Training & Testing Tool started.")
        print("Terminal output will appear both here and in the UI.")
        
        # 应用程序退出代码由splash screen完成窗口显示
        sys.exit(app.exec_())
    except Exception as e:
        # 获取完整的堆栈跟踪
        error_msg = traceback.format_exc()
        
        # 打印到控制台
        print(f"Critical error occurred:\n{error_msg}")
        
        # 显示错误对话框
        try:
            ErrorManager.show_error("Application Error", f"A critical error occurred: {str(e)}", error_msg)
        except Exception:
            # 如果显示错误对话框失败，至少打印到控制台
            print("Failed to show error dialog")
        
        # 错误退出
        sys.exit(1)

if __name__ == "__main__":
    main() 