import os
from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient, QPainterPath, QPen
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt5.QtWidgets import QApplication

class SplashScreen(QSplashScreen):
    def __init__(self):
        # 创建基本pixmap
        pixmap = QPixmap(500, 400)
        pixmap.fill(Qt.transparent)
        super().__init__(pixmap)
        
        # 设置窗口标志
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
        # 创建渐变背景
        self.gradient = QLinearGradient(0, 0, 0, 400)
        self.gradient.setColorAt(0, QColor(16, 24, 32))  # 深蓝色
        self.gradient.setColorAt(1, QColor(30, 40, 52))  # 稍浅的蓝色
        
        # 设置进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 340, 400, 20)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # 隐藏文本
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #34465A;
                border-radius: 10px;
                background-color: rgba(21, 30, 40, 0.7);
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00BFFF, stop:1 #007ACC);
                border-radius: 8px;
            }
        """)
        
        # 创建状态标签
        self.status_label = QLabel(self)
        self.status_label.setGeometry(50, 310, 400, 20)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #DCE6F0;
                font-size: 12px;
                font-weight: 500;
            }
        """)
        
        # 设置标题文本
        self.setFont(QFont('Segoe UI', 14, QFont.Bold))
        
        # 创建动画
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def drawContents(self, painter):
        # 绘制渐变背景
        painter.fillRect(self.rect(), self.gradient)
        
        # 绘制装饰性圆环
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(0, 191, 255, 30), 2))
        painter.drawEllipse(QRect(50, 50, 400, 400))
        painter.drawEllipse(QRect(100, 100, 300, 300))
        
        # 绘制标题
        painter.setPen(QColor(220, 230, 240))
        painter.setFont(QFont('Segoe UI', 24, QFont.Bold))
        painter.drawText(QRect(0, 100, self.width(), 60), 
                        Qt.AlignCenter, 
                        "YOLO DMW")
        
        # 绘制副标题
        painter.setFont(QFont('Segoe UI', 12))
        painter.setPen(QColor(180, 190, 200))
        painter.drawText(QRect(0, 160, self.width(), 30), 
                        Qt.AlignCenter, 
                        "简单快捷的一站式YOLO训练工具")
        
        # 绘制版本信息
        painter.setFont(QFont('Segoe UI', 10))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(QRect(self.width() - 100, self.height() - 30, 80, 20),
                        Qt.AlignRight | Qt.AlignVCenter,
                        "v1.0.0")
        
    def updateProgress(self, value, message=""):
        """更新进度条和消息"""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
            
    def showEvent(self, event):
        """显示事件处理"""
        super().showEvent(event)
        # 设置初始位置（屏幕中央）
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
        
        # 启动淡入动画
        self.animation.setStartValue(QPoint(x, y - 20))
        self.animation.setEndValue(QPoint(x, y))
        self.animation.start()
            
def showSplashScreen(app, main_window):
    """显示启动画面并初始化应用程序"""
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    
    # 模拟加载过程
    steps = [
        (10, "正在初始化应用..."),
        (30, "正在加载组件..."),
        (50, "正在检查GPU..."),
        (70, "正在加载模型..."),
        (90, "正在准备界面..."),
        (100, "启动完成")
    ]
    
    for i, (progress, message) in enumerate(steps):
        QTimer.singleShot(i * 300, lambda p=progress, m=message: splash.updateProgress(p, m))
    
    # 在所有步骤完成后显示主窗口
    QTimer.singleShot(len(steps) * 300 + 200, lambda: finishSplash(splash, main_window))
    
    return splash

def finishSplash(splash, main_window):
    """完成启动画面并显示主窗口"""
    main_window.show()
    splash.finish(main_window) 