import os
import sys
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QComboBox, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, 
                            QMessageBox, QProgressBar, QTextEdit, QScrollArea,
                            QRadioButton, QButtonGroup, QFormLayout, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QColor, QFont, QDesktopServices

from utils.training_worker import TrainingWorker
from utils.data_validator import validate_yolo_dataset, inspect_dataset_structure
from utils.theme_manager import ThemeManager

class TrainingTab(QWidget):
    """Tab for YOLO model training configuration and execution."""
    
    def __init__(self):
        super().__init__()
        self.is_training = False
        self.training_worker = None
        self.training_thread = None
        
        # Default settings
        self.dataset_format = "YOLO"  # Default dataset format
        self.model_type = "yolov8n"  # Default YOLO model
        self.train_mode = "pretrained"  # Default to using pretrained weights
        
        # Default paths (will be updated from settings if available)
        self.default_train_dir = ""
        self.default_val_dir = ""
        self.default_output_dir = ""
        self.default_model_path = ""  # Added for default model path
        
        # Create scroll area with improved settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)  # 移除边框
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 12px;
            }
            QScrollBar:horizontal {
                height: 12px;
            }
        """)
        
        # Create container widget for scroll area
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(15)  # 增加组件之间的间距
        main_layout.setContentsMargins(5, 5, 5, 5)  # 设置边距
        
        # Set up UI
        self.setup_ui(main_layout)
        
        # Set container as scroll area widget
        scroll.setWidget(container)
        
        # Create layout for this widget and add scroll area
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        
        # Initialize UI states
        self.update_fine_tuning_state()
        self.update_weights_path_state()
        
    def setup_ui(self, main_layout):
        """Create and arrange UI elements."""
        # Data section
        data_group = QGroupBox("数据集")
        data_layout = QFormLayout()
        data_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        # Training images and labels
        self.train_images_layout = QHBoxLayout()
        self.train_images_edit = QLineEdit()
        self.train_images_edit.setReadOnly(True)
        self.train_images_btn = QPushButton("浏览...")
        self.train_images_layout.addWidget(self.train_images_edit)
        self.train_images_layout.addWidget(self.train_images_btn)
        data_layout.addRow("训练图像目录:", self.train_images_layout)

        self.train_labels_layout = QHBoxLayout()
        self.train_labels_edit = QLineEdit()
        self.train_labels_edit.setReadOnly(True)
        self.train_labels_btn = QPushButton("浏览...")
        self.train_labels_layout.addWidget(self.train_labels_edit)
        self.train_labels_layout.addWidget(self.train_labels_btn)
        data_layout.addRow("训练标签目录:", self.train_labels_layout)

        # Validation images and labels
        self.val_images_layout = QHBoxLayout()
        self.val_images_edit = QLineEdit()
        self.val_images_edit.setReadOnly(True)
        self.val_images_btn = QPushButton("浏览...")
        self.val_images_layout.addWidget(self.val_images_edit)
        self.val_images_layout.addWidget(self.val_images_btn)
        data_layout.addRow("验证图像目录:", self.val_images_layout)

        self.val_labels_layout = QHBoxLayout()
        self.val_labels_edit = QLineEdit()
        self.val_labels_edit.setReadOnly(True)
        self.val_labels_btn = QPushButton("浏览...")
        self.val_labels_layout.addWidget(self.val_labels_edit)
        self.val_labels_layout.addWidget(self.val_labels_btn)
        data_layout.addRow("验证标签目录:", self.val_labels_layout)
        
        # Add widgets to form layout
        data_group.setLayout(data_layout)
        
        # Model section
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        # Model type selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
                                  "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
                                  "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
                                  "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb"])
        
        # Model Initialization Options
        init_group_box = QGroupBox("模型初始化")
        init_layout = QVBoxLayout()
        
        # Radio buttons for initialization options
        self.model_init_group = QButtonGroup(self)
        
        self.use_pretrained_radio = QRadioButton("使用预训练权重")
        self.from_scratch_radio = QRadioButton("从头开始训练（不使用预训练权重）")
        self.custom_weights_radio = QRadioButton("使用自定义权重")
        
        self.model_init_group.addButton(self.use_pretrained_radio)
        self.model_init_group.addButton(self.from_scratch_radio)
        self.model_init_group.addButton(self.custom_weights_radio)
        
        init_layout.addWidget(self.use_pretrained_radio)
        init_layout.addWidget(self.from_scratch_radio)
        init_layout.addWidget(self.custom_weights_radio)
        
        # Custom weights path layout
        self.model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_btn = QPushButton("浏览...")
        self.model_path_layout.addWidget(self.model_path_edit)
        self.model_path_layout.addWidget(self.model_path_btn)
        
        # Default to use pretrained
        self.use_pretrained_radio.setChecked(True)
        
        # Initially disable the model path controls since 'Use pretrained' is selected by default
        self.model_path_edit.setEnabled(False)
        self.model_path_btn.setEnabled(False)
        
        # Fine-tuning mode
        self.fine_tuning_mode = QCheckBox("微调模式（冻结骨干网络，仅训练检测头）")
        self.fine_tuning_mode.setChecked(False)
        
        init_layout.addWidget(self.fine_tuning_mode)
        init_group_box.setLayout(init_layout)
        
        # Hyperparameters
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(16)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.001)
        
        # Add widgets to form layout
        model_layout.addRow("模型:", self.model_combo)
        model_layout.addRow("自定义权重:", self.model_path_layout)
        model_layout.addRow("批次大小:", self.batch_size_spin)
        model_layout.addRow("训练轮数:", self.epochs_spin)
        model_layout.addRow("图像尺寸:", self.img_size_spin)
        model_layout.addRow("学习率:", self.lr_spin)
        model_layout.addWidget(init_group_box)
        model_group.setLayout(model_layout)
        
        # Output section
        output_group = QGroupBox("输出")
        output_layout = QFormLayout()
        
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        
        self.project_name_edit = QLineEdit("yolo_project")
        
        # Add widgets to form layout
        output_layout.addRow("输出目录:", self.output_dir_layout)
        output_layout.addRow("项目名称:", self.project_name_edit)
        output_group.setLayout(output_layout)
        
        # Control section
        control_layout = QHBoxLayout()
        self.validate_btn = QPushButton("验证数据")
        self.validate_btn.setMinimumHeight(40)
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMinimumHeight(40)
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.validate_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        # Progress section
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setAcceptRichText(True)
        self.log_text.document().setDefaultStyleSheet("pre {margin: 0; padding: 0;}")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_text)
        progress_group.setLayout(progress_layout)
        
        # Add all sections to main layout with proper spacing
        main_layout.addWidget(data_group)
        main_layout.addSpacing(10)
        main_layout.addWidget(model_group)
        main_layout.addSpacing(10)
        main_layout.addWidget(output_group)
        main_layout.addSpacing(10)
        main_layout.addLayout(control_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(progress_group)
        
        # Add stretch at the end to push everything up
        main_layout.addStretch()
        
        # Connect signals
        self.connect_signals()
        
        # For first initialization
        self.model_combo.setCurrentIndex(0)
        self.update_parameters_display()
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        # Model selection change
        self.model_combo.currentIndexChanged.connect(self.update_parameters_display)
        
        # Directory selection
        self.train_images_btn.clicked.connect(lambda: self.select_directory("训练图像目录", self.train_images_edit))
        self.train_labels_btn.clicked.connect(lambda: self.select_directory("训练标签目录", self.train_labels_edit))
        self.val_images_btn.clicked.connect(lambda: self.select_directory("验证图像目录", self.val_images_edit))
        self.val_labels_btn.clicked.connect(lambda: self.select_directory("验证标签目录", self.val_labels_edit))
        self.output_dir_btn.clicked.connect(lambda: self.select_directory("输出目录", self.output_dir_edit))
        self.model_path_btn.clicked.connect(self.select_model_path)
        
        # Form controls
        self.use_pretrained_radio.toggled.connect(self.on_initialization_mode_changed)
        self.custom_weights_radio.toggled.connect(self.on_initialization_mode_changed)
        self.from_scratch_radio.toggled.connect(self.on_initialization_mode_changed)
        
        # Connect fine-tuning checkbox to ensure it works only with pretrained weights
        self.fine_tuning_mode.toggled.connect(self.update_fine_tuning_state)
        
        # Control buttons
        self.validate_btn.clicked.connect(self.validate_dataset)
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
    
    def on_initialization_mode_changed(self, checked=None):
        """Handle changes to initialization mode radio buttons"""
        # Update dependent UI states
        self.update_weights_path_state()
        self.update_fine_tuning_state()
        
        # If training from scratch is selected, disable fine-tuning
        if self.from_scratch_radio.isChecked() and self.fine_tuning_mode.isChecked():
            self.fine_tuning_mode.setChecked(False)
            self.log_message("从头开始训练不支持微调模式，已禁用微调")
    
    def select_train_dir(self):
        """Open dialog to select training data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if dir_path:
            self.train_dir_edit.setText(dir_path)
    
    def select_val_dir(self):
        """Open dialog to select validation data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Validation Data Directory")
        if dir_path:
            self.val_dir_edit.setText(dir_path)
    
    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def select_model_path(self):
        """Open dialog to select model weights file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights", "", "Model Files (*.pt *.pth *.weights);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            # Update fine-tuning state since a model has been selected
            self.update_fine_tuning_state()
    
    def start_training(self):
        """Validate inputs and start training in a separate thread."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable UI elements during training
        self.set_ui_enabled(False)
        self.is_training = True
        
        # Clear log
        self.log_text.clear()
        self.log_message("正在准备训练任务...")
        
        # Determine which training mode to use
        if self.custom_weights_radio.isChecked():
            # Custom weights mode
            model_weights = self.model_path_edit.text()
            if not model_weights:
                self.log_message("警告: 选择了自定义权重，但未指定权重文件")
                QMessageBox.warning(self, "缺少输入", "使用自定义权重模式时，请选择模型权重文件。")
                self.set_ui_enabled(True)
                self.is_training = False
                return
            pretrained = False
        elif self.use_pretrained_radio.isChecked():
            # Pretrained weights mode
            model_weights = None
            pretrained = True
            self.log_message("使用预训练权重初始化模型")
        else:
            # Train from scratch mode
            model_weights = None
            pretrained = False
            self.log_message("从头开始训练模型（不使用预训练权重）")
        
        # Check if it's fine-tuning mode
        fine_tuning = self.fine_tuning_mode.isChecked()
        if fine_tuning and not (self.use_pretrained_radio.isChecked() or model_weights):
            self.log_message("警告: 微调模式需要预训练模型或自定义权重！已禁用微调")
            fine_tuning = False
        
        # Create worker instance
        self.training_worker = TrainingWorker(
            model_type=self.model_combo.currentText(),
            train_dir=self.train_images_edit.text(),
            val_dir=self.val_images_edit.text(),
            output_dir=self.output_dir_edit.text(),
            project_name=self.project_name_edit.text(),
            dataset_format=self.dataset_format,
            batch_size=self.batch_size_spin.value(),
            epochs=self.epochs_spin.value(),
            img_size=self.img_size_spin.value(),
            learning_rate=self.lr_spin.value(),
            pretrained=pretrained,
            model_weights=model_weights,
            fine_tuning=fine_tuning
        )
        
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)
        
        # Connect signals
        self.training_worker.progress_update.connect(self.update_progress)
        self.training_worker.log_update.connect(self.log_message)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.training_error.connect(self.on_training_error)
        self.training_thread.started.connect(self.training_worker.run)
        
        # Start training
        self.training_thread.start()
    
    def stop_training(self):
        """Stop the training process."""
        self.log_message("Stopping training (please wait)...")
        if self.training_worker:
            self.training_worker.stop()
    
    def on_training_complete(self):
        """Handle training completion."""
        self.is_training = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        self.log_message("训练成功完成！")
        QMessageBox.information(self, "训练完成", "训练已成功完成。")
    
    def on_training_error(self, error_msg):
        """Handle training error."""
        self.is_training = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        self.log_message(f"错误: {error_msg}")
        QMessageBox.critical(self, "训练错误", f"训练过程中发生错误:\n{error_msg}")
    
    def clean_up_thread(self):
        """Clean up thread and worker resources."""
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait()
            self.training_thread = None
            self.training_worker = None
    
    def update_progress(self, progress):
        """Update progress bar."""
        self.progress_bar.setValue(progress)
    
    def log_message(self, message):
        """Add a message to the log display."""
        # 检查消息类型
        if "Epoch" in message and ("GPU_mem" in message or "box_loss" in message):
            # 如果是训练进度信息，使用特殊格式
            self.log_text.append(f"\n<span style='color:#0066CC; font-family:Courier;'>{message}</span>")
            # 确保光标可见
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.End)
            self.log_text.setTextCursor(cursor)
            # 自动滚动到底部
            self.log_text.ensureCursorVisible()
        else:
            # 普通消息
            self.log_text.append(message)
        
        # Also print to stdout for terminal redirection
        print(f"[Training] {message}")
    
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during training."""
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.train_images_btn.setEnabled(enabled)
        self.val_images_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        
        # Model path button should only be enabled if custom weights is checked
        model_path_enabled = enabled and self.custom_weights_radio.isChecked()
        self.model_path_btn.setEnabled(model_path_enabled)
        self.model_path_edit.setEnabled(model_path_enabled)
        
        self.model_combo.setEnabled(enabled)
        self.batch_size_spin.setEnabled(enabled)
        self.epochs_spin.setEnabled(enabled)
        self.img_size_spin.setEnabled(enabled)
        self.lr_spin.setEnabled(enabled)
        
        # Radio buttons for model initialization
        self.use_pretrained_radio.setEnabled(enabled)
        self.from_scratch_radio.setEnabled(enabled)
        self.custom_weights_radio.setEnabled(enabled)
        
        # Fine-tuning is only enabled if using pretrained or custom weights
        fine_tuning_enabled = enabled and (self.use_pretrained_radio.isChecked() or self.custom_weights_radio.isChecked())
        self.fine_tuning_mode.setEnabled(fine_tuning_enabled)
        
        self.project_name_edit.setEnabled(enabled)
    
    def validate_inputs(self):
        """Validate user inputs before starting training."""
        # 检查训练图像和标签目录
        if not self.train_images_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择训练图像目录。")
            return False
        if not self.train_labels_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择训练标签目录。")
            return False
        # 检查验证图像和标签目录
        if not self.val_images_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择验证图像目录。")
            return False
        if not self.val_labels_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择验证标签目录。")
            return False
        # 检查输出目录
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择输出目录。")
            return False
        return True
    
    def update_settings(self, settings):
        """Update tab settings based on settings from settings tab."""
        if 'default_model' in settings:
            index = self.model_combo.findText(settings['default_model'])
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                self.model_type = settings['default_model']
        
        if 'default_batch_size' in settings:
            self.batch_size_spin.setValue(settings['default_batch_size'])
        
        if 'default_img_size' in settings:
            self.img_size_spin.setValue(settings['default_img_size'])
        
        # Update default paths for images/labels and set in UI if empty
        if 'default_train_images_dir' in settings:
            if settings['default_train_images_dir'] and not self.train_images_edit.text():
                self.train_images_edit.setText(settings['default_train_images_dir'])
        if 'default_train_labels_dir' in settings:
            if settings['default_train_labels_dir'] and not self.train_labels_edit.text():
                self.train_labels_edit.setText(settings['default_train_labels_dir'])
        if 'default_val_images_dir' in settings:
            if settings['default_val_images_dir'] and not self.val_images_edit.text():
                self.val_images_edit.setText(settings['default_val_images_dir'])
        if 'default_val_labels_dir' in settings:
            if settings['default_val_labels_dir'] and not self.val_labels_edit.text():
                self.val_labels_edit.setText(settings['default_val_labels_dir'])
        
        if 'default_output_dir' in settings:
            self.default_output_dir = settings['default_output_dir']
            if self.default_output_dir and not self.output_dir_edit.text():
                self.output_dir_edit.setText(self.default_output_dir)
            
        # Update default model path and set in UI if empty
        if 'default_train_model_path' in settings:
            self.default_model_path = settings['default_train_model_path']
            if self.default_model_path and not self.model_path_edit.text():
                self.model_path_edit.setText(self.default_model_path)
                # If a default model path is provided, select custom weights radio button
                self.custom_weights_radio.setChecked(True)
            
        # Update UI states
        self.update_weights_path_state()
        self.update_fine_tuning_state()
    
    def update_fine_tuning_state(self, checked=None):
        """Update UI state based on fine-tuning and model initialization options."""
        # Fine-tuning requires pretrained or custom weights
        using_pretrained_weights = self.use_pretrained_radio.isChecked() or self.custom_weights_radio.isChecked()
        using_custom_weights = self.custom_weights_radio.isChecked() and bool(self.model_path_edit.text())
        
        # Fine-tuning is only enabled if using pretrained or valid custom weights
        fine_tuning_enabled = using_pretrained_weights or using_custom_weights
        
        # Set the enabled state of fine_tuning_mode checkbox
        self.fine_tuning_mode.setEnabled(fine_tuning_enabled)
        
        # If fine-tuning is checked but not using pretrained or custom weights, uncheck it
        if self.fine_tuning_mode.isChecked() and not fine_tuning_enabled:
            self.fine_tuning_mode.setChecked(False)
        
        # Update the tooltip based on state
        if fine_tuning_enabled:
            self.fine_tuning_mode.setToolTip("冻结检测头之前的所有参数，仅更新检测头参数")
        else:
            self.fine_tuning_mode.setToolTip("微调模式需要预训练模型或指定的权重文件")
    
    def update_weights_path_state(self, checked=None):
        """Enable or disable custom weights path based on radio button state"""
        # Only enable the model path controls when custom weights is selected
        is_custom = self.custom_weights_radio.isChecked()
        self.model_path_edit.setEnabled(is_custom)
        self.model_path_btn.setEnabled(is_custom)
        
        # Clear the path if custom weights is not selected
        if not is_custom:
            self.model_path_edit.clear()
    
    def select_directory(self, title, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, title)
        if dir_path:
            line_edit.setText(dir_path)
            # 自动同步到设置页
            from ui.main_window import MainWindow
            main_window = self.parentWidget()
            while main_window and not isinstance(main_window, MainWindow):
                main_window = main_window.parentWidget()
            if main_window and hasattr(main_window, 'settings_tab'):
                settings_tab = main_window.settings_tab
                # 根据line_edit对象同步到对应设置项
                if line_edit is self.train_images_edit:
                    settings_tab.default_train_images_edit.setText(dir_path)
                elif line_edit is self.train_labels_edit:
                    settings_tab.default_train_labels_edit.setText(dir_path)
                elif line_edit is self.val_images_edit:
                    settings_tab.default_val_images_edit.setText(dir_path)
                elif line_edit is self.val_labels_edit:
                    settings_tab.default_val_labels_edit.setText(dir_path)
                # 实时保存
                settings_tab.save_settings()
    
    def validate_dataset(self):
        """Validate the dataset structure and image-label matching."""
        # Get the directory paths
        train_images_dir = self.train_images_edit.text()
        train_labels_dir = self.train_labels_edit.text()
        val_images_dir = self.val_images_edit.text()
        val_labels_dir = self.val_labels_edit.text()
        
        # Check if paths are provided
        if not train_images_dir:
            QMessageBox.warning(self, "缺少路径", "请先选择训练图像目录")
            return
            
        # Validate the training dataset
        self.log_message("开始验证训练数据集...")
        train_results = validate_yolo_dataset(train_images_dir, train_labels_dir)
        
        # Log the training validation results
        self.log_message(f"训练数据集验证: {train_results['message']}")
        
        # If validation dataset is provided, validate it as well
        if val_images_dir:
            self.log_message("开始验证验证数据集...")
            val_results = validate_yolo_dataset(val_images_dir, val_labels_dir)
            self.log_message(f"验证数据集验证: {val_results['message']}")
        
        # Inspect dataset structure for more detailed information
        if train_images_dir:
            base_dir = os.path.dirname(os.path.dirname(train_images_dir))
            structure_report = inspect_dataset_structure(base_dir)
            self.log_message("\n数据集结构分析:\n" + structure_report)
        
        # Show summary result
        if train_results["success"] and (not val_images_dir or (val_images_dir and val_results["success"])):
            QMessageBox.information(self, "验证成功", "数据集结构验证通过，可以开始训练。\n\n详细信息已添加到日志。")
        else:
            validation_message = (
                f"数据集验证发现以下问题:\n\n"
                f"训练集: 找到 {train_results['total_images']} 张图片, 匹配 {train_results['matched_labels']} 个标签\n"
            )
            
            if val_images_dir:
                validation_message += f"验证集: 找到 {val_results['total_images']} 张图片, 匹配 {val_results['matched_labels']} 个标签\n\n"
            
            validation_message += "详细信息已添加到日志。请修正问题后重试。"
            
            QMessageBox.warning(self, "验证问题", validation_message)
    
    def update_parameters_display(self):
        """Update UI parameters based on selected model."""
        model = self.model_combo.currentText()
        
        # Adjust batch size based on model size
        if model.endswith('n'):  # nano models
            self.batch_size_spin.setValue(16)
        elif model.endswith('s'):  # small models
            self.batch_size_spin.setValue(16)
        elif model.endswith('m'):  # medium models
            self.batch_size_spin.setValue(8)
        elif model.endswith('l'):  # large models
            self.batch_size_spin.setValue(8)
        elif model.endswith('x'):  # extra large models
            self.batch_size_spin.setValue(4)
        
        # Log model change
        self.log_message(f"已选择模型: {model}")
        
        # Update fine-tuning state in case model changed
        self.update_fine_tuning_state() 