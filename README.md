# 🔍 YOLO目标检测训练与测试工具

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0+-green.svg)](https://ultralytics.com)
[![PyQt5](https://img.shields.io/badge/PyQt-5.15.6-41cd52.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

<div align="center">
<p>🚀 一站式YOLO目标检测解决方案 | 训练、测试、推理、数据集转换一体化工具</p>
</div>

## 📋 功能概览

YOLO目标检测训练与测试工具是一个基于PyQt5的现代化图形用户界面应用程序，提供了使用YOLOv8、YOLOv5和YOLO11系列模型进行目标检测的完整解决方案。

### 🎯 核心特性

- 🔄 支持多种YOLO模型系列（YOLOv8/v5/11/12）
- 💻 现代化图形界面，支持多主题切换
- 🚄 GPU加速支持，实时训练监控
- 📊 实时可视化检测结果
- 🛠️ 灵活的数据集格式转换工具

## 🌟 主要功能

### 1. 🎓 模型训练

<details>
<summary>点击展开详情</summary>

- **数据集管理**：支持YOLO格式的数据集，包括训练和验证数据集的设置
- **模型选择**：支持多种YOLO模型（YOLOv8、YOLOv5、YOLO11、YOLO12系列）
- **初始化选项**：
  - ✨ 使用预训练权重
  - 🆕 从头开始训练
  - 📥 使用自定义权重
- **微调模式**：支持冻结骨干网络，仅训练检测头
- **超参数设置**：
  - 📦 批次大小
  - 🔄 训练轮数
  - 📐 图像尺寸
  - 📈 学习率
- **训练进度监控**：实时进度条和日志输出
![0b9ede1b7c841c778ba8e8ebdc6d302](https://github.com/user-attachments/assets/0a335484-06fa-4a52-92ae-44b18d4cfb9c)

</details>

### 2. 📊 模型测试

<details>
<summary>点击展开详情</summary>

- **模型评估**：对训练好的模型进行精度评估
- **测试数据设置**：指定测试图像和标签目录
- **参数配置**：
  - 🎯 置信度阈值
  - 🔍 IoU阈值
  - 📏 图像尺寸
- **实时结果预览**：显示检测结果图像
- **结果保存**：将测试结果保存到指定目录
- **终端日志**：实时输出测试进度和结果
![fc5320e28d3f4f9cb10ad9da6a910b4](https://github.com/user-attachments/assets/dc2b873e-1d48-4bc3-a25f-d427a4f4e10f)
</details>


### 3. 🔍 模型推理

<details>
<summary>点击展开详情</summary>

- **推理模式**：
  - 📸 单张图片推理
  - 📁 文件夹批量推理
- **参数设置**：
  - 🎯 置信度阈值
  - 🔍 IoU阈值
  - 📏 图像尺寸
- **结果展示**：实时预览检测结果
- **图像浏览器**：查看和浏览所有生成的结果图像
- **结果保存**：将推理结果保存到指定目录
![89871482286e06f07fecea0ed8a4773](https://github.com/user-attachments/assets/a2761916-4f7f-43d7-b1f4-d201022ff039)

</details>

### 4. 🔄 数据集转换

<details>
<summary>点击展开详情</summary>

- **支持格式**：COCO和VOC格式转换为YOLO格式
- **转换模式**：
  - 🔄 整体划分：自动划分训练集和验证集
  - ✂️ 指定训练/验证集：手动指定训练和验证数据
- **验证集比例**：可自定义设置验证集占比
- **输出**：生成符合YOLO标准的数据集，包括images、labels和dataset.yaml
![1a956e0736850c6f17fdf5cb9170c05](https://github.com/user-attachments/assets/fcecaa3b-449a-4729-a2aa-ae15de5fbfa3)

</details>

## 🛠️ 安装要求

<details>
<summary>展开查看依赖库</summary>

```bash
# 核心依赖
PyQt5==5.15.6
PyQt5-sip>=12.9.0
PyQt5-Qt5>=5.15.2

# 深度学习框架
torch>=1.10.0
torchvision>=0.11.0
ultralytics>=8.0.0

# 图像处理和工具库
opencv-python>=4.5.5
numpy>=1.21.0
matplotlib>=3.5.1
pycocotools>=2.0.4
PyYAML>=6.0
tqdm>=4.64.0
```

</details>

## 📚 使用指南

### 🚀 快速开始

```bash
# 标准启动
python main.py

# 科技感主题启动
python start_with_tech_theme.py
```

### 💡 详细教程

<details>
<summary>1. 训练模型</summary>

1. 切换到"训练"标签页
2. 设置训练和验证数据集路径
3. 选择YOLO模型类型和初始化模式
4. 配置训练参数（批次大小、轮数、图像尺寸、学习率）
5. 设置输出目录和项目名称
6. 点击"验证数据"确保数据集格式正确
7. 点击"开始训练"启动训练过程
8. 训练日志和进度将实时显示
![37c2bda5de953af99fdebe79e03bb25](https://github.com/user-attachments/assets/77609d7f-8a37-4b71-952d-765484fcdeea)

</details>

<details>
<summary>2. 测试模型</summary>

1. 切换到"测试"标签页
2. 选择要测试的模型文件
3. 设置测试参数（置信度阈值、IoU阈值、图像尺寸）
4. 指定测试图像和标签目录
5. 设置输出目录
6. 点击"开始测试"进行模型评估
7. 测试结果将显示在右侧预览区域和日志中
![fc5320e28d3f4f9cb10ad9da6a910b4](https://github.com/user-attachments/assets/8fe51cab-90dc-45f1-b2ff-edc623b99731)

</details>

<details>
<summary>3. 推理应用</summary>

1. 切换到"推理"标签页
2. 选择推理模式（图片或文件夹）
3. 选择模型文件和配置参数
4. 指定输入图片或文件夹
5. 设置输出目录
6. 点击"开始推理"进行目标检测
7. 结果图像将显示在预览区域
8. 使用图像浏览器查看所有结果
![89871482286e06f07fecea0ed8a4773](https://github.com/user-attachments/assets/f2fcc580-31ed-47eb-826a-bdfebbb32eaf)

</details>

<details>
<summary>4. 数据集转换</summary>

1. 切换到"数据集转换"标签页
2. 选择源数据集格式（COCO或VOC）
3. 选择转换模式
4. 设置输入和输出路径
5. 点击"开始转换"将数据集转换为YOLO格式
![1a956e0736850c6f17fdf5cb9170c05](https://github.com/user-attachments/assets/24d42af1-2e19-4b85-9eba-a1823e312945)

</details>

## ⚠️ 注意事项

- 💻 确保安装了所有依赖库
- 🚀 推荐使用GPU进行训练和推理以获得更佳性能
- 💾 对于大型数据集，请确保有足够的磁盘空间和内存

## 📬 联系方式

如有问题或建议，欢迎联系：

- 📧 Email: zweicumt@163.com
- 💬 技术支持：[创建Issue](https://github.com/your-repo/issues)

---

<div align="center">
<p>用❤️打造 | MIT License</p>
</div> 
