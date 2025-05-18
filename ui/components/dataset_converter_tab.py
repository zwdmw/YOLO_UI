from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QComboBox, QGroupBox,
                             QTextEdit, QMessageBox, QProgressBar, QLineEdit,
                             QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from utils.dataset_converter import DatasetConverter
import os

class DatasetConverterWorker(QThread):
    """Worker thread for dataset conversion"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, output_path, format_type, val_ratio):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.format_type = format_type
        self.val_ratio = val_ratio
        self.converter = DatasetConverter()
    
    def run(self):
        try:
            self.progress.emit(f"开始转换数据集...\n")
            self.progress.emit(f"输入路径: {self.input_path}\n")
            self.progress.emit(f"输出路径: {self.output_path}\n")
            self.progress.emit(f"格式类型: {self.format_type}\n")
            
            # 检查输入路径是否存在
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"输入路径不存在: {self.input_path}")
            
            # 检查输出路径是否可写
            try:
                os.makedirs(self.output_path, exist_ok=True)
                test_file = os.path.join(self.output_path, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                raise PermissionError(f"输出路径不可写: {self.output_path}")
            
            success = self.converter.convert_dataset(
                self.input_path,
                self.output_path,
                self.format_type,
                self.val_ratio
            )
            
            if success:
                self.progress.emit("数据集转换完成！\n")
                self.progress.emit(f"输出目录: {self.output_path}\n")
                self.progress.emit("包含以下内容:\n")
                self.progress.emit("- images/: 图像文件\n")
                self.progress.emit("- labels/: 标注文件\n")
                self.progress.emit("- dataset.yaml: 数据集配置文件\n")
                self.finished.emit(True, "数据集转换成功完成！")
            else:
                self.progress.emit("数据集转换失败！\n")
                self.finished.emit(False, "数据集转换失败，请检查日志获取详细信息。")
                
        except Exception as e:
            self.progress.emit(f"发生错误: {str(e)}\n")
            self.finished.emit(False, f"发生错误: {str(e)}")

class DatasetConverterTab(QWidget):
    """Dataset converter tab for converting between different dataset formats"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Input group
        input_group = QGroupBox("输入设置")
        input_layout = QVBoxLayout()
        
        # Format selection
        format_layout = QHBoxLayout()
        format_label = QLabel("数据集格式:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["COCO", "VOC"])
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        input_layout.addLayout(format_layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("转换模式:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["整体划分", "指定训练/验证集"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        input_layout.addLayout(mode_layout)
        
        # Overall mode settings
        self.overall_group = QGroupBox("整体划分设置")
        overall_layout = QVBoxLayout()
        
        # Input paths selection for overall mode
        # Images directory
        images_dir_layout = QHBoxLayout()
        self.images_dir_label = QLabel("图像目录:")
        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setReadOnly(True)
        self.images_dir_button = QPushButton("浏览...")
        self.images_dir_button.clicked.connect(lambda: self.select_directory("图像目录", self.images_dir_edit))
        images_dir_layout.addWidget(self.images_dir_label)
        images_dir_layout.addWidget(self.images_dir_edit)
        images_dir_layout.addWidget(self.images_dir_button)
        overall_layout.addLayout(images_dir_layout)
        
        # Labels directory
        labels_dir_layout = QHBoxLayout()
        self.labels_dir_label = QLabel("标签目录:")
        self.labels_dir_edit = QLineEdit()
        self.labels_dir_edit.setReadOnly(True)
        self.labels_dir_button = QPushButton("浏览...")
        self.labels_dir_button.clicked.connect(lambda: self.select_directory("标签目录", self.labels_dir_edit))
        labels_dir_layout.addWidget(self.labels_dir_label)
        labels_dir_layout.addWidget(self.labels_dir_edit)
        labels_dir_layout.addWidget(self.labels_dir_button)
        overall_layout.addLayout(labels_dir_layout)
        
        # Validation ratio
        val_ratio_layout = QHBoxLayout()
        val_ratio_label = QLabel("验证集比例:")
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 0.5)
        self.val_ratio_spin.setValue(0.2)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setDecimals(2)
        val_ratio_layout.addWidget(val_ratio_label)
        val_ratio_layout.addWidget(self.val_ratio_spin)
        overall_layout.addLayout(val_ratio_layout)
        
        self.overall_group.setLayout(overall_layout)
        input_layout.addWidget(self.overall_group)
        
        # Split mode settings
        self.split_group = QGroupBox("训练/验证集设置")
        split_layout = QVBoxLayout()
        
        # Training set paths
        train_layout = QVBoxLayout()
        train_images_layout = QHBoxLayout()
        train_images_label = QLabel("训练图像目录:")
        self.train_images_edit = QLineEdit()
        self.train_images_edit.setReadOnly(True)
        self.train_images_button = QPushButton("浏览...")
        self.train_images_button.clicked.connect(lambda: self.select_directory("训练图像目录", self.train_images_edit))
        train_images_layout.addWidget(train_images_label)
        train_images_layout.addWidget(self.train_images_edit)
        train_images_layout.addWidget(self.train_images_button)
        train_layout.addLayout(train_images_layout)
        
        train_labels_layout = QHBoxLayout()
        train_labels_label = QLabel("训练标签目录:")
        self.train_labels_edit = QLineEdit()
        self.train_labels_edit.setReadOnly(True)
        self.train_labels_button = QPushButton("浏览...")
        self.train_labels_button.clicked.connect(lambda: self.select_directory("训练标签目录", self.train_labels_edit))
        train_labels_layout.addWidget(train_labels_label)
        train_labels_layout.addWidget(self.train_labels_edit)
        train_labels_layout.addWidget(self.train_labels_button)
        train_layout.addLayout(train_labels_layout)
        
        # Validation set paths
        val_layout = QVBoxLayout()
        val_images_layout = QHBoxLayout()
        val_images_label = QLabel("验证图像目录:")
        self.val_images_edit = QLineEdit()
        self.val_images_edit.setReadOnly(True)
        self.val_images_button = QPushButton("浏览...")
        self.val_images_button.clicked.connect(lambda: self.select_directory("验证图像目录", self.val_images_edit))
        val_images_layout.addWidget(val_images_label)
        val_images_layout.addWidget(self.val_images_edit)
        val_images_layout.addWidget(self.val_images_button)
        val_layout.addLayout(val_images_layout)
        
        val_labels_layout = QHBoxLayout()
        val_labels_label = QLabel("验证标签目录:")
        self.val_labels_edit = QLineEdit()
        self.val_labels_edit.setReadOnly(True)
        self.val_labels_button = QPushButton("浏览...")
        self.val_labels_button.clicked.connect(lambda: self.select_directory("验证标签目录", self.val_labels_edit))
        val_labels_layout.addWidget(val_labels_label)
        val_labels_layout.addWidget(self.val_labels_edit)
        val_labels_layout.addWidget(self.val_labels_button)
        val_layout.addLayout(val_labels_layout)
        
        split_layout.addLayout(train_layout)
        split_layout.addLayout(val_layout)
        self.split_group.setLayout(split_layout)
        input_layout.addWidget(self.split_group)
        
        # Output path selection
        output_path_layout = QHBoxLayout()
        output_path_label = QLabel("输出路径:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_button = QPushButton("浏览...")
        self.output_path_button.clicked.connect(self.select_output_path)
        output_path_layout.addWidget(output_path_label)
        output_path_layout.addWidget(self.output_path_edit)
        output_path_layout.addWidget(self.output_path_button)
        input_layout.addLayout(output_path_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Convert button
        self.convert_button = QPushButton("开始转换")
        self.convert_button.clicked.connect(self.start_conversion)
        layout.addWidget(self.convert_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        
        # Initialize UI state
        self.on_mode_changed(self.mode_combo.currentText())
    
    def on_format_changed(self, format_type):
        """Handle format selection change"""
        self.input_path_edit.clear()
        self.log_text.clear()
    
    def on_mode_changed(self, mode):
        """Handle mode selection change"""
        if mode == "整体划分":
            self.overall_group.setVisible(True)
            self.split_group.setVisible(False)
        else:  # 指定训练/验证集
            self.overall_group.setVisible(False)
            self.split_group.setVisible(True)
        self.log_text.append(f"已选择转换模式: {mode}\n")
    
    def select_input_path(self):
        """Select input dataset path"""
        format_type = self.format_combo.currentText().lower()
        if format_type == "coco":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择COCO标注文件", "", "JSON Files (*.json)"
            )
            if file_path:
                self.input_path_edit.setText(file_path)
                self.log_text.append(f"已选择COCO标注文件: {file_path}\n")
        else:  # VOC format
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择VOC数据集目录"
            )
            if dir_path:
                self.input_path_edit.setText(dir_path)
                self.log_text.append(f"已选择VOC数据集目录: {dir_path}\n")
    
    def select_directory(self, title, line_edit):
        """Select directory and update line edit"""
        dir_path = QFileDialog.getExistingDirectory(
            self, f"选择{title}"
        )
        if dir_path:
            line_edit.setText(dir_path)
            self.log_text.append(f"已选择{title}: {dir_path}\n")
    
    def select_output_path(self):
        """Select output dataset path"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录"
        )
        if dir_path:
            self.output_path_edit.setText(dir_path)
            self.log_text.append(f"已选择输出目录: {dir_path}\n")
    
    def start_conversion(self):
        """Start dataset conversion process"""
        try:
            # Get conversion parameters
            format_type = self.format_combo.currentText().lower()
            mode = "overall" if self.mode_combo.currentText() == "整体划分" else "split"
            output_path = self.output_path_edit.text()
            
            if not output_path:
                QMessageBox.warning(self, "错误", "请选择输出目录")
                return
            
            # Create converter instance
            converter = DatasetConverter()
            
            # Start conversion based on mode
            if mode == "overall":
                images_dir = self.images_dir_edit.text()
                labels_dir = self.labels_dir_edit.text()
                
                if not images_dir:
                    QMessageBox.warning(self, "错误", "请选择图像目录")
                    return
                
                if not labels_dir:
                    QMessageBox.warning(self, "错误", "请选择标签目录")
                    return
                
                val_ratio = self.val_ratio_spin.value()
                success = converter.convert_dataset(
                    input_path=labels_dir,  # Use labels directory as input path
                    output_path=output_path,
                    format_type=format_type,
                    mode=mode,
                    val_ratio=val_ratio,
                    train_images_dir=images_dir  # Use images directory for images
                )
            else:  # split mode
                train_images_dir = self.train_images_edit.text()
                train_labels_dir = self.train_labels_edit.text()
                val_images_dir = self.val_images_edit.text()
                val_labels_dir = self.val_labels_edit.text()
                
                if not all([train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]):
                    QMessageBox.warning(self, "错误", "请选择所有训练和验证集目录")
                    return
                
                success = converter.convert_dataset(
                    input_path=train_labels_dir,  # Use train labels dir as input path
                    output_path=output_path,
                    format_type=format_type,
                    mode=mode,
                    train_images_dir=train_images_dir,
                    train_labels_dir=train_labels_dir,
                    val_images_dir=val_images_dir,
                    val_labels_dir=val_labels_dir
                )
            
            if success:
                QMessageBox.information(self, "成功", "数据集转换完成")
                self.log_text.append("数据集转换完成\n")
            else:
                QMessageBox.warning(self, "错误", "数据集转换失败")
                self.log_text.append("数据集转换失败\n")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"转换过程中发生错误: {str(e)}")
            self.log_text.append(f"错误: {str(e)}\n") 