import os
import sys
from PyQt5.QtGui import QPainter, QColor, QPixmap, QLinearGradient, QFont, QRadialGradient, QPen, QIcon
from PyQt5.QtCore import Qt, QSize, QPoint, QRect

def create_app_icon():
    """创建应用程序主图标"""
    # 创建一个128x128的图标
    pixmap = QPixmap(128, 128)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制渐变背景
    gradient = QRadialGradient(64, 64, 64)
    gradient.setColorAt(0, QColor(0, 160, 233))  # 科技蓝色
    gradient.setColorAt(1, QColor(16, 24, 32))   # 深蓝色
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(4, 4, 120, 120)
    
    # 绘制Y字母
    font = QFont("Arial", 72, QFont.Bold)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(QRect(0, 0, 128, 128), Qt.AlignCenter, "Y")
    
    # 绘制外圈
    pen = QPen(QColor(0, 191, 255), 3)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(6, 6, 116, 116)
    
    painter.end()
    
    # 保存图标
    icon_path = "app_icon.png"
    pixmap.save(icon_path)
    return icon_path

def create_train_icon():
    """创建训练标签页图标"""
    # 创建一个64x64的图标
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制背景
    gradient = QLinearGradient(0, 0, 64, 64)
    gradient.setColorAt(0, QColor(0, 120, 212))
    gradient.setColorAt(1, QColor(0, 80, 160))
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
    
    # 绘制图形（数据线和上升箭头）
    pen = QPen(QColor(255, 255, 255), 2)
    painter.setPen(pen)
    
    # 绘制折线图样式
    points = [(10, 45), (20, 35), (30, 40), (40, 25), (50, 15)]
    for i in range(len(points) - 1):
        painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    
    # 在线的端点绘制小圆点
    painter.setBrush(QColor(255, 255, 255))
    for point in points:
        painter.drawEllipse(point[0] - 2, point[1] - 2, 4, 4)
    
    painter.end()
    
    # 保存图标
    icon_path = "train_icon.png"
    pixmap.save(icon_path)
    return icon_path

def create_test_icon():
    """创建测试标签页图标"""
    # 创建一个64x64的图标
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制背景
    gradient = QLinearGradient(0, 0, 64, 64)
    gradient.setColorAt(0, QColor(0, 150, 136))  # 科技绿色
    gradient.setColorAt(1, QColor(0, 105, 92))   
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
    
    # 绘制测试图标（对勾符号）
    pen = QPen(QColor(255, 255, 255), 3)
    painter.setPen(pen)
    
    # 对勾路径
    points = [(15, 32), (25, 42), (45, 22)]
    painter.drawLine(points[0][0], points[0][1], points[1][0], points[1][1])
    painter.drawLine(points[1][0], points[1][1], points[2][0], points[2][1])
    
    # 绘制圆形边框
    painter.setPen(QPen(QColor(255, 255, 255, 150), 2))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(12, 12, 40, 40)
    
    painter.end()
    
    # 保存图标
    icon_path = "test_icon.png"
    pixmap.save(icon_path)
    return icon_path

def create_inference_icon():
    """创建推理标签页图标"""
    # 创建一个64x64的图标
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制背景
    gradient = QLinearGradient(0, 0, 64, 64)
    gradient.setColorAt(0, QColor(156, 39, 176))  # 紫色
    gradient.setColorAt(1, QColor(103, 58, 183))  # 偏蓝紫
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
    
    # 绘制大脑或AI图标
    pen = QPen(QColor(255, 255, 255), 2)
    painter.setPen(pen)
    
    # 绘制一个简化的大脑形状
    painter.setBrush(Qt.NoBrush)
    
    # 大脑轮廓
    painter.drawEllipse(20, 15, 24, 30)
    
    # 连接线条，模拟神经网络
    painter.drawLine(20, 32, 15, 32)
    painter.drawLine(44, 32, 49, 32)
    painter.drawLine(32, 15, 32, 10)
    painter.drawLine(32, 45, 32, 50)
    
    # 绘制一些连接点
    painter.setBrush(QColor(255, 255, 255))
    painter.drawEllipse(13, 30, 4, 4)
    painter.drawEllipse(47, 30, 4, 4)
    painter.drawEllipse(30, 8, 4, 4)
    painter.drawEllipse(30, 48, 4, 4)
    
    # 内部连接线
    painter.drawLine(20, 25, 44, 25)
    painter.drawLine(20, 35, 44, 35)
    painter.drawLine(27, 15, 27, 45)
    painter.drawLine(37, 15, 37, 45)
    
    painter.end()
    
    # 保存图标
    icon_path = "inference_icon.png"
    pixmap.save(icon_path)
    return icon_path

def create_settings_icon():
    """创建设置标签页图标"""
    # 创建一个64x64的图标
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制背景
    gradient = QLinearGradient(0, 0, 64, 64)
    gradient.setColorAt(0, QColor(96, 125, 139))  # 蓝灰色
    gradient.setColorAt(1, QColor(69, 90, 100))
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
    
    # 绘制齿轮
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    
    # 外圆
    painter.drawEllipse(24, 24, 16, 16)
    
    # 齿轮的齿
    for i in range(8):
        painter.save()
        painter.translate(32, 32)
        painter.rotate(i * 45)
        painter.drawRoundedRect(-2, -20, 4, 10, 2, 2)
        painter.restore()
    
    # 内圆（空心）
    painter.setBrush(gradient)
    painter.drawEllipse(28, 28, 8, 8)
    
    painter.end()
    
    # 保存图标
    icon_path = "settings_icon.png"
    pixmap.save(icon_path)
    return icon_path

def generate_icons():
    """生成所有图标并保存到当前目录"""
    print("开始生成图标...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到当前目录
    os.chdir(current_dir)
    
    # 生成图标
    app_icon = create_app_icon()
    train_icon = create_train_icon()
    test_icon = create_test_icon()
    inference_icon = create_inference_icon()
    settings_icon = create_settings_icon()
    
    print(f"图标已生成：")
    print(f"- {app_icon}")
    print(f"- {train_icon}")
    print(f"- {test_icon}")
    print(f"- {inference_icon}")
    print(f"- {settings_icon}")
    
    return 0

if __name__ == "__main__":
    sys.exit(generate_icons()) 