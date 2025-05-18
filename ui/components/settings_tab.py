import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QLineEdit, QSpinBox, 
                            QDoubleSpinBox, QGroupBox, QCheckBox, QMessageBox,
                            QFormLayout, QGridLayout, QTabWidget, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from utils.theme_manager import ThemeManager

class SettingsTab(QWidget):
    """Tab for application settings and preferences."""
    
    # Signal to notify other tabs when settings are updated
    settings_updated = pyqtSignal(dict)
    theme_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Initialize default settings
        self.settings = {
            'default_model': 'yolov8n',
            'default_batch_size': 16,
            'default_img_size': 640,
            'default_conf_thresh': 0.25,
            'default_iou_thresh': 0.45,
            'use_gpu': True,
            'gpu_device': 0,
            'default_train_dir': '',
            'default_val_dir': '',
            'default_test_dir': '',
            'default_output_dir': '',
            'default_train_model_path': '',
            'default_test_model_path': '',
            'theme': 'tech',  # 默认主题为科技感主题
            'default_train_images_dir': '',
            'default_train_labels_dir': '',
            'default_val_images_dir': '',
            'default_val_labels_dir': '',
            'default_test_images_dir': '',
            'default_test_labels_dir': ''
        }
        
        # Load settings if available
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'settings.json')
        self.load_settings()
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Create and arrange UI elements."""
        main_layout = QVBoxLayout(self)
        
        # Create tabs for different settings categories
        settings_tabs = QTabWidget()
        
        # General settings tab
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        
        # Hardware settings
        hardware_group = QGroupBox("硬件设置")
        hardware_layout = QFormLayout()
        
        self.use_gpu_check = QCheckBox("使用GPU")
        self.use_gpu_check.setChecked(self.settings['use_gpu'])
        
        self.gpu_device_spin = QSpinBox()
        self.gpu_device_spin.setRange(0, 8)
        self.gpu_device_spin.setValue(self.settings['gpu_device'])
        self.gpu_device_spin.setEnabled(self.settings['use_gpu'])
        
        hardware_layout.addRow("", self.use_gpu_check)
        hardware_layout.addRow("GPU设备:", self.gpu_device_spin)
        hardware_group.setLayout(hardware_layout)
        
        # Default model settings
        model_group = QGroupBox("默认模型设置")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
                                  "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
                                  "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"])
        index = self.model_combo.findText(self.settings['default_model'])
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        
        model_layout.addRow("默认模型:", self.model_combo)
        model_group.setLayout(model_layout)

        # UI settings group
        ui_group = QGroupBox("界面设置")
        ui_layout = QFormLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色主题", "深色主题", "科技感主题"])
        
        # 根据设置确定当前主题索引
        if self.settings['theme'] == 'light':
            theme_index = 0
        elif self.settings['theme'] == 'dark':
            theme_index = 1
        else:  # tech theme
            theme_index = 2
            
        self.theme_combo.setCurrentIndex(theme_index)
        
        ui_layout.addRow("主题:", self.theme_combo)
        ui_group.setLayout(ui_layout)
        
        # Add groups to general tab
        general_layout.addWidget(hardware_group)
        general_layout.addWidget(model_group)
        general_layout.addWidget(ui_group)
        
        # Path settings tab
        paths_tab = QWidget()
        paths_layout = QFormLayout(paths_tab)
        
        # Default path settings
        paths_group = QGroupBox("默认目录")
        paths_form_layout = QFormLayout()
        
        # Training data path
        self.train_dir_layout = QHBoxLayout()
        self.train_dir_edit = QLineEdit(self.settings['default_train_dir'])
        self.train_dir_edit.setReadOnly(True)
        self.train_dir_btn = QPushButton("浏览...")
        self.train_dir_layout.addWidget(self.train_dir_edit)
        self.train_dir_layout.addWidget(self.train_dir_btn)
        
        # Validation data path
        self.val_dir_layout = QHBoxLayout()
        self.val_dir_edit = QLineEdit(self.settings['default_val_dir'])
        self.val_dir_edit.setReadOnly(True)
        self.val_dir_btn = QPushButton("浏览...")
        self.val_dir_layout.addWidget(self.val_dir_edit)
        self.val_dir_layout.addWidget(self.val_dir_btn)
        
        # Test data path
        self.test_dir_layout = QHBoxLayout()
        self.test_dir_edit = QLineEdit(self.settings['default_test_dir'])
        self.test_dir_edit.setReadOnly(True)
        self.test_dir_btn = QPushButton("浏览...")
        self.test_dir_layout.addWidget(self.test_dir_edit)
        self.test_dir_layout.addWidget(self.test_dir_btn)
        
        # Output path
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.settings['default_output_dir'])
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        
        # Training model path
        self.train_model_layout = QHBoxLayout()
        self.train_model_edit = QLineEdit(self.settings['default_train_model_path'])
        self.train_model_edit.setReadOnly(True)
        self.train_model_btn = QPushButton("浏览...")
        self.train_model_layout.addWidget(self.train_model_edit)
        self.train_model_layout.addWidget(self.train_model_btn)
        
        # Testing model path
        self.test_model_layout = QHBoxLayout()
        self.test_model_edit = QLineEdit(self.settings['default_test_model_path'])
        self.test_model_edit.setReadOnly(True)
        self.test_model_btn = QPushButton("浏览...")
        self.test_model_layout.addWidget(self.test_model_edit)
        self.test_model_layout.addWidget(self.test_model_btn)
        
        # 新增：默认训练图像目录
        self.default_train_images_layout = QHBoxLayout()
        self.default_train_images_edit = QLineEdit(self.settings.get('default_train_images_dir', ''))
        self.default_train_images_edit.setReadOnly(True)
        self.default_train_images_btn = QPushButton("浏览...")
        self.default_train_images_layout.addWidget(self.default_train_images_edit)
        self.default_train_images_layout.addWidget(self.default_train_images_btn)
        paths_form_layout.addRow("默认训练图像目录:", self.default_train_images_layout)

        # 新增：默认训练标签目录
        self.default_train_labels_layout = QHBoxLayout()
        self.default_train_labels_edit = QLineEdit(self.settings.get('default_train_labels_dir', ''))
        self.default_train_labels_edit.setReadOnly(True)
        self.default_train_labels_btn = QPushButton("浏览...")
        self.default_train_labels_layout.addWidget(self.default_train_labels_edit)
        self.default_train_labels_layout.addWidget(self.default_train_labels_btn)
        paths_form_layout.addRow("默认训练标签目录:", self.default_train_labels_layout)

        # 新增：默认验证图像目录
        self.default_val_images_layout = QHBoxLayout()
        self.default_val_images_edit = QLineEdit(self.settings.get('default_val_images_dir', ''))
        self.default_val_images_edit.setReadOnly(True)
        self.default_val_images_btn = QPushButton("浏览...")
        self.default_val_images_layout.addWidget(self.default_val_images_edit)
        self.default_val_images_layout.addWidget(self.default_val_images_btn)
        paths_form_layout.addRow("默认验证图像目录:", self.default_val_images_layout)

        # 新增：默认验证标签目录
        self.default_val_labels_layout = QHBoxLayout()
        self.default_val_labels_edit = QLineEdit(self.settings.get('default_val_labels_dir', ''))
        self.default_val_labels_edit.setReadOnly(True)
        self.default_val_labels_btn = QPushButton("浏览...")
        self.default_val_labels_layout.addWidget(self.default_val_labels_edit)
        self.default_val_labels_layout.addWidget(self.default_val_labels_btn)
        paths_form_layout.addRow("默认验证标签目录:", self.default_val_labels_layout)

        # 新增：默认测试图像目录
        self.default_test_images_layout = QHBoxLayout()
        self.default_test_images_edit = QLineEdit(self.settings.get('default_test_images_dir', ''))
        self.default_test_images_edit.setReadOnly(True)
        self.default_test_images_btn = QPushButton("浏览...")
        self.default_test_images_layout.addWidget(self.default_test_images_edit)
        self.default_test_images_layout.addWidget(self.default_test_images_btn)
        paths_form_layout.addRow("默认测试图像目录:", self.default_test_images_layout)

        # 新增：默认测试标签目录
        self.default_test_labels_layout = QHBoxLayout()
        self.default_test_labels_edit = QLineEdit(self.settings.get('default_test_labels_dir', ''))
        self.default_test_labels_edit.setReadOnly(True)
        self.default_test_labels_btn = QPushButton("浏览...")
        self.default_test_labels_layout.addWidget(self.default_test_labels_edit)
        self.default_test_labels_layout.addWidget(self.default_test_labels_btn)
        paths_form_layout.addRow("默认测试标签目录:", self.default_test_labels_layout)
        
        paths_form_layout.addRow("输出目录:", self.output_dir_layout)
        paths_form_layout.addRow("训练模型:", self.train_model_layout)
        paths_form_layout.addRow("测试模型:", self.test_model_layout)
        
        paths_group.setLayout(paths_form_layout)
        paths_layout.addWidget(paths_group)
        
        # Training settings tab
        training_tab = QWidget()
        training_layout = QFormLayout(training_tab)
        
        # Default training parameters
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(self.settings['default_batch_size'])
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(self.settings['default_img_size'])
        self.img_size_spin.setSingleStep(32)
        
        training_layout.addRow("默认批次大小:", self.batch_size_spin)
        training_layout.addRow("默认图像尺寸:", self.img_size_spin)
        
        # Testing settings tab
        testing_tab = QWidget()
        testing_layout = QFormLayout(testing_tab)
        
        # Default testing parameters
        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.1, 1.0)
        self.conf_thresh_spin.setValue(self.settings['default_conf_thresh'])
        self.conf_thresh_spin.setSingleStep(0.05)
        
        self.iou_thresh_spin = QDoubleSpinBox()
        self.iou_thresh_spin.setRange(0.1, 1.0)
        self.iou_thresh_spin.setValue(self.settings['default_iou_thresh'])
        self.iou_thresh_spin.setSingleStep(0.05)
        
        testing_layout.addRow("默认置信度阈值:", self.conf_thresh_spin)
        testing_layout.addRow("默认IoU阈值:", self.iou_thresh_spin)
        
        # Add tabs to the settings tab widget
        settings_tabs.addTab(general_tab, "通用")
        settings_tabs.addTab(paths_tab, "路径")
        settings_tabs.addTab(training_tab, "训练")
        settings_tabs.addTab(testing_tab, "测试")
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("保存设置")
        self.save_btn.setMinimumHeight(40)
        
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        
        # Add widgets to main layout
        main_layout.addWidget(settings_tabs)
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        self.save_btn.clicked.connect(self.save_settings)
        self.reset_btn.clicked.connect(self.reset_settings)
        self.use_gpu_check.toggled.connect(self.toggle_gpu_settings)
        self.theme_combo.currentIndexChanged.connect(self.apply_theme)
        
        # Connect path selection buttons
        self.train_dir_btn.clicked.connect(self.select_train_dir)
        self.val_dir_btn.clicked.connect(self.select_val_dir)
        self.test_dir_btn.clicked.connect(self.select_test_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        self.train_model_btn.clicked.connect(self.select_train_model)
        self.test_model_btn.clicked.connect(self.select_test_model)
        
        # 新增：默认训练图像目录
        self.default_train_images_btn.clicked.connect(lambda: self.select_directory(self.default_train_images_edit, "选择默认训练图像目录"))
        self.default_train_labels_btn.clicked.connect(lambda: self.select_directory(self.default_train_labels_edit, "选择默认训练标签目录"))
        self.default_val_images_btn.clicked.connect(lambda: self.select_directory(self.default_val_images_edit, "选择默认验证图像目录"))
        self.default_val_labels_btn.clicked.connect(lambda: self.select_directory(self.default_val_labels_edit, "选择默认验证标签目录"))
        self.default_test_images_btn.clicked.connect(lambda: self.select_directory(self.default_test_images_edit, "选择默认测试图像目录"))
        self.default_test_labels_btn.clicked.connect(lambda: self.select_directory(self.default_test_labels_edit, "选择默认测试标签目录"))
    
    def toggle_gpu_settings(self, checked):
        """Enable or disable GPU-related settings based on checkbox."""
        self.gpu_device_spin.setEnabled(checked)
    
    def apply_theme(self, index):
        """应用选中的主题"""
        app = QApplication.instance()
        if index == 0:  # 浅色主题
            ThemeManager.apply_light_theme(app)
            self.settings['theme'] = 'light'
        elif index == 1:  # 深色主题
            ThemeManager.apply_dark_theme(app)
            self.settings['theme'] = 'dark'
        else:  # tech theme
            ThemeManager.apply_tech_theme(app)
            self.settings['theme'] = 'tech'
        
        # 发送主题已更改的信号
        self.theme_changed.emit(self.settings['theme'])
    
    def select_train_dir(self):
        """Open dialog to select training data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认训练数据目录")
        if dir_path:
            self.train_dir_edit.setText(dir_path)
    
    def select_val_dir(self):
        """Open dialog to select validation data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认验证数据目录")
        if dir_path:
            self.val_dir_edit.setText(dir_path)
    
    def select_test_dir(self):
        """Open dialog to select test data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认测试数据目录")
        if dir_path:
            self.test_dir_edit.setText(dir_path)
    
    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def select_train_model(self):
        """Open dialog to select training model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择默认训练模型", "", "模型文件 (*.pt *.pth *.weights);;所有文件 (*)"
        )
        if file_path:
            self.train_model_edit.setText(file_path)
    
    def select_test_model(self):
        """Open dialog to select testing model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择默认测试模型", "", "模型文件 (*.pt *.pth *.weights);;所有文件 (*)"
        )
        if file_path:
            self.test_model_edit.setText(file_path)
    
    def select_directory(self, line_edit, title):
        dir_path = QFileDialog.getExistingDirectory(self, title)
        if dir_path:
            line_edit.setText(dir_path)
    
    def save_settings(self):
        """Save current settings to file and emit signal to update other tabs."""
        # Update settings dict from UI
        self.settings['default_model'] = self.model_combo.currentText()
        self.settings['default_batch_size'] = self.batch_size_spin.value()
        self.settings['default_img_size'] = self.img_size_spin.value()
        self.settings['default_conf_thresh'] = self.conf_thresh_spin.value()
        self.settings['default_iou_thresh'] = self.iou_thresh_spin.value()
        self.settings['use_gpu'] = self.use_gpu_check.isChecked()
        self.settings['gpu_device'] = self.gpu_device_spin.value()
        
        # 更新主题设置
        theme_index = self.theme_combo.currentIndex()
        if theme_index == 0:
            self.settings['theme'] = 'light'
        elif theme_index == 1:
            self.settings['theme'] = 'dark' 
        else:
            self.settings['theme'] = 'tech'
        
        # Update path settings
        self.settings['default_train_dir'] = self.train_dir_edit.text()
        self.settings['default_val_dir'] = self.val_dir_edit.text()
        self.settings['default_test_dir'] = self.test_dir_edit.text()
        self.settings['default_output_dir'] = self.output_dir_edit.text()
        self.settings['default_train_model_path'] = self.train_model_edit.text()
        self.settings['default_test_model_path'] = self.test_model_edit.text()
        
        # 新增：默认训练图像目录
        self.settings['default_train_images_dir'] = self.default_train_images_edit.text()
        self.settings['default_train_labels_dir'] = self.default_train_labels_edit.text()
        self.settings['default_val_images_dir'] = self.default_val_images_edit.text()
        self.settings['default_val_labels_dir'] = self.default_val_labels_edit.text()
        self.settings['default_test_images_dir'] = self.default_test_images_edit.text()
        self.settings['default_test_labels_dir'] = self.default_test_labels_edit.text()
        
        # Save to file
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            
            # Notify other tabs
            self.settings_updated.emit(self.settings)
            
            QMessageBox.information(self, "设置已保存", "设置已成功保存。")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存设置失败: {str(e)}")
    
    def load_settings(self):
        """Load settings from file if it exists."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                
                # Update settings with loaded values
                for key, value in loaded_settings.items():
                    if key in self.settings:
                        self.settings[key] = value
                
                # Update UI
                self.model_combo.setCurrentText(self.settings['default_model'])
                self.batch_size_spin.setValue(self.settings['default_batch_size'])
                self.img_size_spin.setValue(self.settings['default_img_size'])
                self.conf_thresh_spin.setValue(self.settings['default_conf_thresh'])
                self.iou_thresh_spin.setValue(self.settings['default_iou_thresh'])
                self.use_gpu_check.setChecked(self.settings['use_gpu'])
                self.gpu_device_spin.setValue(self.settings['gpu_device'])
                self.train_dir_edit.setText(self.settings['default_train_dir'])
                self.val_dir_edit.setText(self.settings['default_val_dir'])
                self.test_dir_edit.setText(self.settings['default_test_dir'])
                self.output_dir_edit.setText(self.settings['default_output_dir'])
                self.train_model_edit.setText(self.settings['default_train_model_path'])
                self.test_model_edit.setText(self.settings['default_test_model_path'])
                self.theme_combo.setCurrentIndex(2)  # 设置为科技感主题
                
                # 应用默认主题
                app = QApplication.instance()
                ThemeManager.apply_tech_theme(app)
                
                # 新增：默认训练图像目录
                self.default_train_images_edit.setText(self.settings.get('default_train_images_dir', ''))
                self.default_train_labels_edit.setText(self.settings.get('default_train_labels_dir', ''))
                self.default_val_images_edit.setText(self.settings.get('default_val_images_dir', ''))
                self.default_val_labels_edit.setText(self.settings.get('default_val_labels_dir', ''))
                self.default_test_images_edit.setText(self.settings.get('default_test_images_dir', ''))
                self.default_test_labels_edit.setText(self.settings.get('default_test_labels_dir', ''))
                
                # Save to file and notify other tabs
                self.save_settings()
            
            except Exception as e:
                print(f"加载设置时出错: {str(e)}")
    
    def reset_settings(self):
        """Reset settings to default values."""
        reply = QMessageBox.question(
            self, '确认重置',
            "你确定要将所有设置重置为默认值吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset to default values
            self.settings = {
                'default_model': 'yolov8n',
                'default_batch_size': 16,
                'default_img_size': 640,
                'default_conf_thresh': 0.25,
                'default_iou_thresh': 0.45,
                'use_gpu': True,
                'gpu_device': 0,
                'default_train_dir': '',
                'default_val_dir': '',
                'default_test_dir': '',
                'default_output_dir': '',
                'default_train_model_path': '',
                'default_test_model_path': '',
                'theme': 'tech',  # 默认重置为科技感主题
                'default_train_images_dir': '',
                'default_train_labels_dir': '',
                'default_val_images_dir': '',
                'default_val_labels_dir': '',
                'default_test_images_dir': '',
                'default_test_labels_dir': ''
            }
            
            # Update UI
            self.model_combo.setCurrentText(self.settings['default_model'])
            self.batch_size_spin.setValue(self.settings['default_batch_size'])
            self.img_size_spin.setValue(self.settings['default_img_size'])
            self.conf_thresh_spin.setValue(self.settings['default_conf_thresh'])
            self.iou_thresh_spin.setValue(self.settings['default_iou_thresh'])
            self.use_gpu_check.setChecked(self.settings['use_gpu'])
            self.gpu_device_spin.setValue(self.settings['gpu_device'])
            self.train_dir_edit.setText(self.settings['default_train_dir'])
            self.val_dir_edit.setText(self.settings['default_val_dir'])
            self.test_dir_edit.setText(self.settings['default_test_dir'])
            self.output_dir_edit.setText(self.settings['default_output_dir'])
            self.train_model_edit.setText(self.settings['default_train_model_path'])
            self.test_model_edit.setText(self.settings['default_test_model_path'])
            self.theme_combo.setCurrentIndex(2)  # 设置为科技感主题
            
            # 应用默认主题
            app = QApplication.instance()
            ThemeManager.apply_tech_theme(app)
            
            # 新增：默认训练图像目录
            self.default_train_images_edit.setText(self.settings['default_train_images_dir'])
            self.default_train_labels_edit.setText(self.settings['default_train_labels_dir'])
            self.default_val_images_edit.setText(self.settings['default_val_images_dir'])
            self.default_val_labels_edit.setText(self.settings['default_val_labels_dir'])
            self.default_test_images_edit.setText(self.settings['default_test_images_dir'])
            self.default_test_labels_edit.setText(self.settings['default_test_labels_dir'])
            
            # Save to file and notify other tabs
            self.save_settings()