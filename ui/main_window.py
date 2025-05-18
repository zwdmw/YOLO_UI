import os
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QCheckBox, QMessageBox, QProgressBar,
                             QTextEdit, QRadioButton, QButtonGroup, QSplitter, QApplication,
                             QStyleFactory)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap
from PyQt5.QtSvg import QSvgRenderer

from ui.components.training_tab import TrainingTab
from ui.components.testing_tab import TestingTab
from ui.components.settings_tab import SettingsTab
from ui.components.inference_tab import InferenceTab
from ui.components.dataset_converter_tab import DatasetConverterTab
from utils.terminal_redirect import TerminalManager
from utils.theme_manager import ThemeManager

class MainWindow(QMainWindow):
    """Main application window containing all interface elements."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO目标检测训练与测试工具 zweicumt@163.com")
        self.setMinimumSize(1200, 900)  # 增加最小尺寸
        self.resize(1400, 1000)  # 设置默认尺寸
        
        # 设置应用图标
        self.set_app_icon()
        
        # 初始化主题管理器
        ThemeManager.initialize()
        
        # 创建设置标签页（需要先创建以便获取主题设置）
        self.settings_tab = SettingsTab()
        
        # 立即应用保存的主题设置 - 提前应用主题以便其他组件使用正确的样式
        self.apply_saved_theme()
        
        # Initialize terminal redirection
        self.terminal_manager = TerminalManager()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create other tabs
        self.training_tab = TrainingTab()
        self.testing_tab = TestingTab()
        self.inference_tab = InferenceTab()
        self.dataset_converter_tab = DatasetConverterTab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.training_tab, "训练")
        self.tab_widget.addTab(self.testing_tab, "测试")
        self.tab_widget.addTab(self.inference_tab, "推理")
        self.tab_widget.addTab(self.dataset_converter_tab, "数据集转换")
        self.tab_widget.addTab(self.settings_tab, "设置")
        
        # 为标签页添加图标
        self.setup_tab_icons()
        
        # 直接将标签页添加到主布局
        main_layout.addWidget(self.tab_widget)
        
        # Set up signal connections between tabs if needed
        self.setup_connections()
        
        # Apply styling
        self.setup_styling()
        
        # Setup terminal redirection 
        self.setup_terminal_redirection()
        
        # Load default settings and apply to tabs
        self.load_default_settings()
        
        # 主动触发一次主题应用，确保所有组件都应用了正确的主题
        self.force_apply_theme()
    
    def force_apply_theme(self):
        """强制重新应用当前主题"""
        app = QApplication.instance()
        theme = self.settings_tab.settings['theme']
        
        # 使用更强的样式强制应用
        if theme == 'light':
            ThemeManager.apply_light_theme(app)
        elif theme == 'dark':
            ThemeManager.apply_dark_theme(app)
        else:  # 科技感主题
            ThemeManager.apply_tech_theme(app)
        
        # 更新所有标签页
        self.update()
        self.repaint()
        
        # 更新子组件
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).update()
            self.tab_widget.widget(i).repaint()
    
    def set_app_icon(self):
        """设置应用程序图标"""
        # SVG图标路径
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui/assets/app_icon.svg")
        
        try:
            # 从SVG文件创建图标
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            
            # 同时设置应用程序级图标
            app = QApplication.instance()
            app.setWindowIcon(app_icon)
        except Exception as e:
            print(f"设置应用图标时出错: {str(e)}")
    
    def create_icon_from_svg(self, svg_path):
        """从SVG文件创建QIcon"""
        if os.path.exists(svg_path):
            return QIcon(svg_path)
        else:
            print(f"SVG文件不存在: {svg_path}")
            return QIcon()
    
    def setup_tab_icons(self):
        """为标签页设置图标"""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui/assets")
        
        # 设置标签页图标
        try:
            # 使用SVG图标
            train_icon = self.create_icon_from_svg(os.path.join(assets_dir, "train_icon.svg"))
            test_icon = self.create_icon_from_svg(os.path.join(assets_dir, "test_icon.svg"))
            inference_icon = self.create_icon_from_svg(os.path.join(assets_dir, "inference_icon.svg"))
            settings_icon = self.create_icon_from_svg(os.path.join(assets_dir, "settings_icon.svg"))
            converter_icon = self.create_icon_from_svg(os.path.join(assets_dir, "converter_icon.svg"))
            
            self.tab_widget.setTabIcon(0, train_icon)
            self.tab_widget.setTabIcon(1, test_icon)
            self.tab_widget.setTabIcon(2, inference_icon)
            self.tab_widget.setTabIcon(3, converter_icon)
            self.tab_widget.setTabIcon(4, settings_icon)
        except Exception as e:
            print(f"设置标签页图标时出错: {str(e)}")
    
    def setup_connections(self):
        """Set up signal connections between tabs"""
        # Connect settings to training tab
        self.settings_tab.settings_updated.connect(self.training_tab.update_settings)
        self.settings_tab.settings_updated.connect(self.testing_tab.update_settings)
        self.settings_tab.settings_updated.connect(self.inference_tab.update_settings)
        
        # 连接主题变更信号
        self.settings_tab.theme_changed.connect(self.on_theme_changed)
    
    def on_theme_changed(self, theme):
        """处理主题变更"""
        # 主题变更已经由settings_tab中的apply_theme方法处理
        # 主动触发一次额外的主题应用，确保所有组件都使用新主题
        self.force_apply_theme()
        print(f"主题已变更为: {theme}")
    
    def apply_saved_theme(self):
        """应用保存的主题设置"""
        # 获取设置信息
        if 'theme' in self.settings_tab.settings:
            theme = self.settings_tab.settings['theme']
            app = QApplication.instance()
            
            # 应用对应的主题
            if theme == 'light':
                ThemeManager.apply_light_theme(app)
            elif theme == 'dark':
                ThemeManager.apply_dark_theme(app)
            else:  # 科技感主题
                ThemeManager.apply_tech_theme(app)
    
    def setup_styling(self):
        """Apply styling to the UI elements"""
        # Set the font for the entire application
        font = QFont("Segoe UI", 10)
        self.setFont(font)
        
        # Set style for tab widget
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        
        # 增加标签页的图标大小
        self.tab_widget.setIconSize(QSize(24, 24))  # 稍微增大图标以便更好地显示SVG
    
    def setup_terminal_redirection(self):
        """Set up terminal output redirection to the UI."""
        try:
            # Connect to training tab's log text area
            self.terminal_manager.connect_to_text_edit(self.training_tab.log_text)
            
            # Connect to testing tab's log text area
            self.terminal_manager.connect_to_text_edit(self.testing_tab.log_text)
            
            # Also connect to inference tab's log text area
            self.terminal_manager.connect_to_text_edit(self.inference_tab.log_text)
            
            # Connect to dataset converter tab's log text area
            self.terminal_manager.connect_to_text_edit(self.dataset_converter_tab.log_text)
            
            # Start redirection
            self.terminal_manager.start_redirection()
            print("终端输出重定向已初始化")
            print("标准输出和错误将显示在训练、测试、推理和数据集转换页面的终端输出区域")
        except Exception as e:
            import traceback
            print(f"设置终端重定向时出错: {str(e)}")
            print(traceback.format_exc())
    
    def clear_terminal(self):
        """清除终端输出内容"""
        # 这个方法将被测试标签页的清除按钮调用
        # 由于测试标签页已经有自己的clear_terminal方法，这里不需要额外操作
        pass
    
    def closeEvent(self, event):
        """Handle close event - ask for confirmation if training/testing is in progress"""
        if (hasattr(self.training_tab, 'is_training') and self.training_tab.is_training) or \
           (hasattr(self.testing_tab, 'is_testing') and self.testing_tab.is_testing) or \
           (hasattr(self.inference_tab, 'is_inferencing') and self.inference_tab.is_inferencing):
            reply = QMessageBox.question(
                self, '确认退出',
                "有进程正在运行。确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # Stop terminal redirection before closing
                self.terminal_manager.stop_redirection()
                event.accept()
            else:
                event.ignore()
        else:
            # Stop terminal redirection before closing
            self.terminal_manager.stop_redirection()
            event.accept()
    
    def load_default_settings(self):
        """Load default settings from settings file and apply to tabs."""
        try:
            # Get settings from settings tab
            settings = self.settings_tab.settings
            
            # Update training and testing tabs with default settings
            self.training_tab.update_settings(settings)
            self.testing_tab.update_settings(settings)
            self.inference_tab.update_settings(settings)
            
            print("默认设置已加载。")
        except Exception as e:
            import traceback
            print(f"加载默认设置时出错: {str(e)}")
            print(traceback.format_exc()) 