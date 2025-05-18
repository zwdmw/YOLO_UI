import traceback
import sys
from PyQt5.QtWidgets import QMessageBox, QApplication, QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

class ErrorDialog(QDialog):
    """自定义错误对话框，提供更详细的错误信息展示"""
    
    def __init__(self, title, message, details, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self.setup_ui(message, details)
        
    def setup_ui(self, message, details):
        """设置对话框UI"""
        layout = QVBoxLayout(self)
        
        # 错误图标和消息
        error_label = QLabel()
        error_label.setText(f"<h3>❌ {message}</h3>")
        layout.addWidget(error_label)
        
        # 详细信息
        detail_edit = QTextEdit()
        detail_edit.setReadOnly(True)
        detail_edit.setFont(QFont("Courier New", 9))
        detail_edit.setText(details)
        layout.addWidget(detail_edit)
        
        # 帮助提示
        help_label = QLabel("如果问题持续存在，请尝试以下操作:")
        layout.addWidget(help_label)
        
        tips_label = QLabel(
            "1. 检查您的输入参数是否正确<br>"
            "2. 确保您的数据格式符合要求<br>"
            "3. 检查模型文件是否完整<br>"
            "4. 重启应用程序<br>"
            "5. 更新到最新版本"
        )
        layout.addWidget(tips_label)
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

class ErrorManager:
    """错误管理器，用于处理和显示应用程序错误"""
    
    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        """全局异常处理程序"""
        # 获取错误信息
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # 打印到控制台
        print(f"发生未捕获的异常:\n{error_msg}")
        
        # 显示错误对话框
        ErrorManager.show_error("应用程序错误", str(exc_value), error_msg)
        
        # 不继续传播异常（这会终止程序）
        return True
    
    @staticmethod
    def show_error(title, message, details=None):
        """显示错误对话框"""
        try:
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            
            # 如果有详细信息，使用自定义错误对话框
            if details:
                dialog = ErrorDialog(title, message, details)
                dialog.exec_()
            else:
                # 否则使用简单的消息框
                error_box = QMessageBox()
                error_box.setIcon(QMessageBox.Critical)
                error_box.setWindowTitle(title)
                error_box.setText(message)
                error_box.setStandardButtons(QMessageBox.Ok)
                error_box.exec_()
        except Exception as e:
            # 如果显示错误对话框失败，至少打印到控制台
            print(f"显示错误对话框失败: {str(e)}")
            print(f"原始错误: {message}")
            if details:
                print(f"详细信息: {details}")
    
    @staticmethod
    def install_global_handler():
        """安装全局异常处理程序"""
        sys.excepthook = ErrorManager.handle_exception
        
    @staticmethod
    def try_except_decorator(func):
        """装饰器，为函数添加异常处理"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_trace = traceback.format_exc()
                ErrorManager.show_error("函数执行错误", f"{func.__name__} 执行时出错: {str(e)}", error_trace)
                print(f"函数 {func.__name__} 执行时出错: {str(e)}\n{error_trace}")
                return None
        return wrapper 