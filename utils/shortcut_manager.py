from PyQt5.QtWidgets import QShortcut, QApplication
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

class ShortcutManager:
    """管理应用程序快捷键的类"""
    
    def __init__(self, main_window):
        """初始化快捷键管理器"""
        self.main_window = main_window
        self.shortcuts = {}
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """设置常用快捷键"""
        # 设置全局快捷键
        self.add_shortcut("Ctrl+Q", self.main_window.close, "退出应用程序")
        self.add_shortcut("F1", self.show_help, "显示帮助")
        self.add_shortcut("F5", self.refresh_ui, "刷新界面")
        self.add_shortcut("Ctrl+Tab", self.next_tab, "切换到下一个标签页")
        self.add_shortcut("Ctrl+Shift+Tab", self.prev_tab, "切换到上一个标签页")
        
        # 标签页快捷键
        self.add_shortcut("Ctrl+1", lambda: self.set_tab_index(0), "切换到训练标签页")
        self.add_shortcut("Ctrl+2", lambda: self.set_tab_index(1), "切换到测试标签页")
        self.add_shortcut("Ctrl+3", lambda: self.set_tab_index(2), "切换到推理标签页")
        self.add_shortcut("Ctrl+4", lambda: self.set_tab_index(3), "切换到设置标签页")
        
        # 常用操作快捷键
        self.add_shortcut("Ctrl+S", self.save_settings, "保存设置")
        self.add_shortcut("Ctrl+L", self.clear_console, "清除控制台")
        self.add_shortcut("F11", self.toggle_fullscreen, "切换全屏模式")
        
    def add_shortcut(self, key_sequence, callback, description):
        """添加快捷键"""
        shortcut = QShortcut(QKeySequence(key_sequence), self.main_window)
        shortcut.activated.connect(callback)
        self.shortcuts[key_sequence] = {
            "shortcut": shortcut,
            "description": description,
            "callback": callback
        }
    
    def show_help(self):
        """显示帮助对话框，列出所有快捷键"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget, QGridLayout
        
        help_dialog = QDialog(self.main_window)
        help_dialog.setWindowTitle("键盘快捷键")
        help_dialog.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(help_dialog)
        
        # 创建可滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)
        
        # 添加标题
        title = QLabel("<h2>可用的键盘快捷键</h2>")
        scroll_layout.addWidget(title, 0, 0, 1, 2)
        
        # 添加快捷键列表
        row = 1
        for key, info in self.shortcuts.items():
            key_label = QLabel(f"<b>{key}</b>")
            desc_label = QLabel(info["description"])
            scroll_layout.addWidget(key_label, row, 0)
            scroll_layout.addWidget(desc_label, row, 1)
            row += 1
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(help_dialog.accept)
        layout.addWidget(close_button)
        
        help_dialog.exec_()
    
    def refresh_ui(self):
        """刷新UI界面"""
        # 获取当前标签页索引
        current_tab = self.main_window.tab_widget.currentIndex()
        # 刷新当前标签页
        if current_tab == 0:  # 训练标签页
            if hasattr(self.main_window.training_tab, 'refresh'):
                self.main_window.training_tab.refresh()
        elif current_tab == 1:  # 测试标签页
            if hasattr(self.main_window.testing_tab, 'refresh'):
                self.main_window.testing_tab.refresh()
        elif current_tab == 2:  # 推理标签页
            if hasattr(self.main_window.inference_tab, 'refresh'):
                self.main_window.inference_tab.refresh()
        elif current_tab == 3:  # 设置标签页
            if hasattr(self.main_window.settings_tab, 'refresh'):
                self.main_window.settings_tab.refresh()
    
    def next_tab(self):
        """切换到下一个标签页"""
        current = self.main_window.tab_widget.currentIndex()
        count = self.main_window.tab_widget.count()
        self.main_window.tab_widget.setCurrentIndex((current + 1) % count)
    
    def prev_tab(self):
        """切换到上一个标签页"""
        current = self.main_window.tab_widget.currentIndex()
        count = self.main_window.tab_widget.count()
        self.main_window.tab_widget.setCurrentIndex((current - 1) % count)
    
    def set_tab_index(self, index):
        """设置当前标签页索引"""
        if 0 <= index < self.main_window.tab_widget.count():
            self.main_window.tab_widget.setCurrentIndex(index)
    
    def save_settings(self):
        """保存设置"""
        if hasattr(self.main_window.settings_tab, 'save_settings'):
            self.main_window.settings_tab.save_settings()
    
    def clear_console(self):
        """清除控制台输出"""
        current_tab = self.main_window.tab_widget.currentIndex()
        if current_tab == 0 and hasattr(self.main_window.training_tab, 'clear_log'):
            self.main_window.training_tab.clear_log()
        elif current_tab == 1 and hasattr(self.main_window.testing_tab, 'clear_log'):
            self.main_window.testing_tab.clear_log()
        elif current_tab == 2 and hasattr(self.main_window.inference_tab, 'clear_log'):
            self.main_window.inference_tab.clear_log()
    
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.main_window.isFullScreen():
            self.main_window.showNormal()
        else:
            self.main_window.showFullScreen() 