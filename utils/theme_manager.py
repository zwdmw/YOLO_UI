from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor, QFontDatabase, QFont
from PyQt5.QtCore import Qt

class ThemeManager:
    """管理应用程序主题的类，支持浅色、深色和科技感主题模式"""
    
    @staticmethod
    def initialize():
        """初始化主题管理器"""
        # 加载自定义字体（如果有）
        try:
            # fontId = QFontDatabase.addApplicationFont("ui/assets/fonts/Rajdhani-Medium.ttf")
            pass
        except Exception as e:
            print(f"初始化主题管理器时出错: {str(e)}")

    @staticmethod
    def apply_light_theme(app):
        """应用浅色主题到应用程序"""
        app.setStyle('Fusion')
        app.setPalette(QPalette())  # 重置为默认浅色调色板
        
        # 设置主题属性
        app.setProperty('theme', 'light')
        
        # 设置基本样式表
        app.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #F5F5F7;
            }
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
            }
            QToolTip { 
                color: #333333; 
                background-color: #F8F8F8; 
                border: 1px solid #CCCCCC;
                padding: 5px;
                font-size: 11px;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                top: -1px;
            }
            QTabBar::tab {
                background: #F0F0F0;
                border: 1px solid #CCCCCC;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom-color: #FFFFFF;
            }
            QTabBar::tab:hover:!selected {
                background-color: #E6E6E6;
            }
            QPushButton {
                background-color: #F5F5F5;
                border: 1px solid #CCCCCC;
                padding: 6px 12px;
                border-radius: 4px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #E6E6E6;
                border-color: #BBBBBB;
            }
            QPushButton:pressed {
                background-color: #D1D1D1;
            }
            QTextEdit, QLineEdit {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 6px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #99C2FF;
            }
            QComboBox {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px 10px;
                background-color: #FFFFFF;
            }
            QComboBox:hover {
                border-color: #BBBBBB;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #CCCCCC;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #CCCCCC;
                background-color: #FFFFFF;
                selection-background-color: #E6E6E6;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px;
                background-color: #FFFFFF;
            }
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                text-align: center;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #4287f5;
                border-radius: 3px;
            }
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #555555;
            }
            QScrollBar:vertical {
                border: none;
                background: #F0F0F0;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #CCCCCC;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #AAAAAA;
            }
            QScrollBar:horizontal {
                border: none;
                background: #F0F0F0;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #CCCCCC;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #AAAAAA;
            }
        """)
    
    @staticmethod
    def apply_dark_theme(app):
        """应用深色主题到应用程序"""
        app.setStyle('Fusion')
        
        # 设置主题属性
        app.setProperty('theme', 'dark')
        
        # 设置深色调色板
        palette = QPalette()
        # 基础颜色
        palette.setColor(QPalette.Window, QColor(35, 35, 35))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, QColor(40, 40, 40))
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # 禁用状态颜色
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        
        app.setPalette(palette)
        
        # 设置基本样式表
        app.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #1E1E1E;
            }
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
            }
            QToolTip { 
                color: #FFFFFF; 
                background-color: #2D2D30; 
                border: 1px solid #555555;
                padding: 5px;
                font-size: 11px;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                border-radius: 3px;
                top: -1px;
            }
            QTabBar::tab {
                background: #2D2D30;
                border: 1px solid #444444;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #3E3E42;
                border-bottom-color: #3E3E42;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3A3A3E;
            }
            QPushButton {
                background-color: #3E3E42;
                border: 1px solid #555555;
                padding: 6px 12px;
                color: #FFFFFF;
                border-radius: 4px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #505054;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #2A2A2C;
            }
            QTextEdit, QLineEdit {
                background-color: #252526;
                border: 1px solid #3F3F46;
                color: #FFFFFF;
                border-radius: 4px;
                padding: 6px;
            }
            QTextEdit {
                background-color: #1E1E1E;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #3F7FBF;
            }
            QComboBox {
                background-color: #2D2D30;
                border: 1px solid #3F3F46;
                color: #FFFFFF;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QComboBox:hover {
                border-color: #666666;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #3F3F46;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D30;
                border: 1px solid #3F3F46;
                color: #FFFFFF;
                selection-background-color: #3E3E42;
            }
            QProgressBar {
                border: 1px solid #3F3F46;
                border-radius: 4px;
                text-align: center;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                border-radius: 3px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2D2D30;
                border: 1px solid #3F3F46;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 4px;
            }
            QCheckBox {
                color: #FFFFFF;
                spacing: 5px;
            }
            QRadioButton {
                color: #FFFFFF;
                spacing: 5px;
            }
            QGroupBox {
                border: 1px solid #3F3F46;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #CCCCCC;
            }
            QScrollBar:vertical {
                border: none;
                background: #2D2D30;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777777;
            }
            QScrollBar:horizontal {
                border: none;
                background: #2D2D30;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #555555;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #777777;
            }
        """)
    
    @staticmethod
    def apply_tech_theme(app):
        """应用科技感主题到应用程序"""
        app.setStyle('Fusion')
        
        # 设置主题属性
        app.setProperty('theme', 'tech')
        
        # 设置科技感调色板
        palette = QPalette()
        # 基础颜色 - 深蓝色背景
        palette.setColor(QPalette.Window, QColor(16, 24, 32))
        palette.setColor(QPalette.WindowText, QColor(220, 230, 240))
        palette.setColor(QPalette.Base, QColor(21, 30, 40))
        palette.setColor(QPalette.AlternateBase, QColor(30, 40, 52))
        palette.setColor(QPalette.ToolTipBase, QColor(25, 35, 45))
        palette.setColor(QPalette.ToolTipText, QColor(220, 230, 240))
        palette.setColor(QPalette.Text, QColor(220, 230, 240))
        palette.setColor(QPalette.Button, QColor(30, 40, 52))
        palette.setColor(QPalette.ButtonText, QColor(220, 230, 240))
        palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.Link, QColor(42, 170, 218))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # 禁用状态颜色
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 130, 140))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 130, 140))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 130, 140))
        
        app.setPalette(palette)
        
        # 设置科技感样式表
        app.setStyleSheet("""
            QWidget {
                background-color: #101820;
                color: #DCE6F0;
                font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
                font-weight: 400;
            }
            
            QMainWindow, QDialog {
                background-color: #101820;
            }
            
            QToolTip { 
                color: #DCE6F0; 
                background-color: #1E2832; 
                border: 1px solid #34465A;
                padding: 6px;
                font-size: 11px;
                border-radius: 3px;
            }
            
            QTabWidget {
                background-color: transparent;
            }
            
            QTabWidget::pane {
                border: 1px solid #34465A;
                border-radius: 4px;
                top: -1px;
                background-color: rgba(26, 37, 48, 0.95);
            }
            
            QTabBar {
                background-color: transparent;
            }
            
            QTabBar::tab {
                background: #1A2530;
                border: 1px solid #34465A;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                color: #B0C0D0;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2C3E50, stop:1 #223040);
                color: #00BFFF;
                border-bottom-color: #2C3E50;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #243242;
                color: #00BFFF;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2C3E50, stop:1 #223040);
                border: 1px solid #34465A;
                padding: 6px 14px;
                color: #DCE6F0;
                border-radius: 5px;
                min-height: 24px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A4C5E, stop:1 #2C3E50);
                border-color: #00BFFF;
                color: #FFFFFF;
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #223040, stop:1 #1A2530);
                padding-top: 7px;
                padding-bottom: 5px;
            }
            
            QTextEdit, QLineEdit {
                background-color: #151E28;
                border: 1px solid #34465A;
                color: #DCE6F0;
                border-radius: 5px;
                padding: 7px;
                selection-background-color: #00BFFF;
                selection-color: #FFFFFF;
            }
            
            QTextEdit {
                background-color: #121A22;
            }
            
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #00BFFF;
                background-color: #172130;
            }
            
            QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2C3E50, stop:1 #223040);
                border: 1px solid #34465A;
                color: #DCE6F0;
                padding: 5px 10px;
                border-radius: 5px;
                min-height: 24px;
            }
            
            QComboBox:hover {
                border-color: #00BFFF;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #34465A;
            }
            
            QComboBox QAbstractItemView {
                background-color: #1E2832;
                border: 1px solid #34465A;
                color: #DCE6F0;
                selection-background-color: #00BFFF;
                selection-color: #FFFFFF;
            }
            
            QProgressBar {
                border: 1px solid #34465A;
                border-radius: 5px;
                text-align: center;
                height: 16px;
                background-color: #151E28;
                color: #FFFFFF;
            }
            
            QProgressBar::chunk {
                background-color: #00BFFF;
                border-radius: 3px;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #151E28;
                border: 1px solid #34465A;
                color: #DCE6F0;
                padding: 5px;
                border-radius: 5px;
                min-height: 24px;
                selection-background-color: #00BFFF;
                selection-color: #FFFFFF;
            }
            
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #34465A;
                border-bottom: 1px solid #34465A;
                background: #223040;
            }
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #34465A;
                border-top: 1px solid #34465A;
                background: #223040;
            }
            
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: #2C3E50;
            }
            
            QCheckBox {
                color: #DCE6F0;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #34465A;
                border-radius: 3px;
                background: #151E28;
            }
            
            QCheckBox::indicator:checked {
                background-color: #00BFFF;
            }
            
            QRadioButton {
                color: #DCE6F0;
                spacing: 8px;
            }
            
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #34465A;
                border-radius: 9px;
                background: #151E28;
            }
            
            QRadioButton::indicator:checked {
                background-color: #00BFFF;
                width: 10px;
                height: 10px;
                margin: 3px;
            }
            
            QGroupBox {
                border: 1px solid #34465A;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: 500;
                background-color: rgba(30, 40, 50, 0.4);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #00BFFF;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                border: none;
                background: #1A2530;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background: #34465A;
                min-height: 20px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #00BFFF;
            }
            
            QScrollBar:horizontal {
                border: none;
                background: #1A2530;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:horizontal {
                background: #34465A;
                min-width: 20px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background: #00BFFF;
            }
            
            QLabel {
                color: #DCE6F0;
                background: transparent;
            }
            
            QSlider::groove:horizontal {
                height: 5px;
                background: #1A2530;
                border-radius: 2px;
            }
            
            QSlider::handle:horizontal {
                background: #00BFFF;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QMenu {
                background-color: #1E2832;
                border: 1px solid #34465A;
                color: #DCE6F0;
            }
            
            QMenu::item:selected {
                background-color: #00BFFF;
                color: #FFFFFF;
            }
            
            QFrame#line {
                color: #34465A;
            }
            
            QSplitter::handle {
                background-color: #34465A;
            }
            
            QStatusBar {
                background-color: #1A2530;
                color: #B0C0D0;
            }
        """) 