import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QComboBox, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, 
                            QMessageBox, QProgressBar, QTextEdit, QRadioButton,
                            QButtonGroup, QFormLayout, QGridLayout, QSplitter,
                            QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication

from utils.testing_worker import TestingWorker
from utils.data_validator import validate_yolo_dataset, inspect_dataset_structure
from utils.theme_manager import ThemeManager

class TestingTab(QWidget):
    """Tab for YOLO model testing and inference."""
    
    def __init__(self):
        super().__init__()
        self.is_testing = False
        self.testing_worker = None
        self.testing_thread = None
        
        # Default settings
        self.dataset_format = "YOLO"  # Default dataset format
        
        # Default paths (will be updated from settings if available)
        self.default_test_dir = ""
        self.default_output_dir = ""
        self.default_model_path = ""  # Added for default model path
        
        # Set up UI
        self.setup_ui()
        self.apply_theme_styles()  # 初始化时应用主题样式
        
    def setup_ui(self):
        """Create and arrange UI elements."""
        main_layout = QVBoxLayout(self)
        
        # Left Panel - Settings
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        
        # Model section
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
        
        # Model selection
        self.model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_btn = QPushButton("浏览...")
        self.model_path_layout.addWidget(self.model_path_edit)
        self.model_path_layout.addWidget(self.model_path_btn)
        
        # Confidence threshold
        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.1, 1.0)
        self.conf_thresh_spin.setValue(0.25)
        self.conf_thresh_spin.setSingleStep(0.05)
        
        # IoU threshold
        self.iou_thresh_spin = QDoubleSpinBox()
        self.iou_thresh_spin.setRange(0.1, 1.0)
        self.iou_thresh_spin.setValue(0.45)
        self.iou_thresh_spin.setSingleStep(0.05)
        
        # Image size
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        
        # Add widgets to form layout
        model_layout.addRow("模型路径:", self.model_path_layout)
        model_layout.addRow("置信度阈值:", self.conf_thresh_spin)
        model_layout.addRow("IoU阈值:", self.iou_thresh_spin)
        model_layout.addRow("图像尺寸:", self.img_size_spin)
        model_group.setLayout(model_layout)
        
        # Data section
        data_group = QGroupBox("数据设置")
        data_layout = QFormLayout()
        
        # Directory selection
        self.test_images_layout = QHBoxLayout()
        self.test_images_edit = QLineEdit()
        self.test_images_edit.setReadOnly(True)
        self.test_images_btn = QPushButton("浏览...")
        self.test_images_layout.addWidget(self.test_images_edit)
        self.test_images_layout.addWidget(self.test_images_btn)
        data_layout.addRow("测试图像目录:", self.test_images_layout)

        # 新增：测试标签目录
        self.test_labels_layout = QHBoxLayout()
        self.test_labels_edit = QLineEdit()
        self.test_labels_edit.setReadOnly(True)
        self.test_labels_btn = QPushButton("浏览...")
        self.test_labels_layout.addWidget(self.test_labels_edit)
        self.test_labels_layout.addWidget(self.test_labels_btn)
        data_layout.addRow("测试标签目录:", self.test_labels_layout)
        
        # Output directory
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        
        # Add widgets to form layout
        data_layout.addRow("输出目录:", self.output_dir_layout)
        
        self.save_results = QCheckBox("保存检测结果")
        self.save_results.setChecked(True)
        data_layout.addRow("", self.save_results)
        
        data_group.setLayout(data_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.validate_btn = QPushButton("验证数据")
        self.validate_btn.setMinimumHeight(40)
        self.start_btn = QPushButton("开始测试")
        self.start_btn.setMinimumHeight(40)
        self.stop_btn = QPushButton("停止测试")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.validate_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        # Add sections to settings panel
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(data_group)
        settings_layout.addLayout(control_layout)
        
        # 添加信息面板，减少空白区域
        info_group = QGroupBox("任务信息")
        info_layout = QVBoxLayout()
        
        # 添加状态显示
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        info_layout.addWidget(self.status_label)
        
        # 添加简要统计信息
        self.stats_label = QLabel("等待测试开始...")
        info_layout.addWidget(self.stats_label)
        
        # 添加进度指示
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        info_layout.addLayout(progress_layout)
        
        info_group.setLayout(info_layout)
        settings_layout.addWidget(info_group)
        
        # 添加很小的伸缩因子，减少剩余空白
        settings_layout.addStretch(1)
        
        # Right Panel - Results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # 创建一个水平分割器，左侧为设置，右侧为结果
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(settings_panel)
        main_splitter.addWidget(results_panel)
        main_splitter.setSizes([300, 700])  # 调整左右比例，让结果区域占更多空间
        
        # 右侧使用垂直分割器，上部为结果图像，下部为终端输出
        results_splitter = QSplitter(Qt.Vertical)
        
        # 上半部分：结果图像区域
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距使内容更充分利用空间
        
        # 图像预览标题栏改为更明显的样式
        image_header = QWidget()
        image_header_layout = QHBoxLayout(image_header)
        image_header_layout.setContentsMargins(0, 0, 0, 5)  # 减小下边距
        
        image_title = QLabel("检测结果预览")
        image_title.setFont(QFont("", 11, QFont.Bold))  # 增大字体
        image_header_layout.addWidget(image_title)
        
        # 添加保存按钮，样式更明显
        save_img_btn = QPushButton("保存图片")
        save_img_btn.setMaximumWidth(100)  # 稍微加宽按钮
        save_img_btn.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ccc; border-radius: 3px;")
        save_img_btn.clicked.connect(self.save_preview_image)
        image_header_layout.addWidget(save_img_btn, alignment=Qt.AlignRight)
        
        image_layout.addWidget(image_header)
        
        # 图像显示区域
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumHeight(320)  # 略微增加高度
        self.image_display.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;")
        
        # 将图像放入滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setWidget(self.image_display)
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # 移除滚动区域边框
        
        image_layout.addWidget(scroll_area)

        # 下半部分：终端输出区域（铺满下半部分）
        terminal_widget = QWidget()
        terminal_layout = QVBoxLayout(terminal_widget)
        terminal_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        
        # 终端输出标题栏
        terminal_header = QWidget()
        terminal_header_layout = QHBoxLayout(terminal_header)
        terminal_header_layout.setContentsMargins(0, 0, 0, 5)  # 减小下边距
        
        terminal_title = QLabel("终端输出")
        terminal_title.setFont(QFont("", 11, QFont.Bold))  # 增大字体
        terminal_header_layout.addWidget(terminal_title)
        
        # 添加清除按钮
        clear_btn = QPushButton("清除")
        clear_btn.setMaximumWidth(80)
        clear_btn.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ccc; border-radius: 3px;")
        clear_btn.clicked.connect(self.clear_terminal)
        terminal_header_layout.addWidget(clear_btn, alignment=Qt.AlignRight)
        
        terminal_layout.addWidget(terminal_header)
        
        
        # 终端输出文本区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)  # 不自动换行
        self.log_text.document().setDefaultStyleSheet("pre {margin: 0; font-family: monospace;}")
        self.log_text.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ddd; border-radius: 4px; font-family: Consolas, monospace;")
        terminal_layout.addWidget(self.log_text)
        
        # 添加组件到结果分割器
        results_splitter.addWidget(image_widget)
        results_splitter.addWidget(terminal_widget)
        results_splitter.setSizes([300, 500])  # 调整比例，让终端输出占更多空间
        
        # 将分割器添加到结果面板
        results_layout.addWidget(results_splitter)
        
        # 将主分割器添加到主布局
        main_layout.addWidget(main_splitter)
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        # Model selection
        self.model_path_btn.clicked.connect(self.select_model_path)
        
        # Directory selection
        self.test_images_btn.clicked.connect(lambda: self.select_directory(self.test_images_edit, "选择测试图像目录"))
        self.test_labels_btn.clicked.connect(lambda: self.select_directory(self.test_labels_edit, "选择测试标签目录"))
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        
        # Control buttons
        self.validate_btn.clicked.connect(self.validate_dataset)
        self.start_btn.clicked.connect(self.start_testing)
        self.stop_btn.clicked.connect(self.stop_testing)
        
        # 监听主题切换信号
        from ui.main_window import MainWindow
        main_window = self.parentWidget()
        while main_window and not isinstance(main_window, MainWindow):
            main_window = main_window.parentWidget()
        if main_window and hasattr(main_window, 'settings_tab'):
            main_window.settings_tab.theme_changed.connect(lambda _: self.apply_theme_styles())
    
    def select_model_path(self):
        """Open dialog to select model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pt *.pth *.weights);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def select_directory(self, line_edit, title):
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
                if line_edit is self.test_images_edit:
                    settings_tab.default_test_images_edit.setText(dir_path)
                elif line_edit is self.test_labels_edit:
                    settings_tab.default_test_labels_edit.setText(dir_path)
                # 实时保存
                settings_tab.save_settings()
    
    def start_testing(self):
        """Validate inputs and start testing in a separate thread."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable UI elements during testing
        self.set_ui_enabled(False)
        self.is_testing = True
        
        # 重置状态和进度
        self.status_label.setText("状态: 准备中")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        self.progress_bar.setValue(0)
        self.stats_label.setText("正在加载模型...")
        
        # Clear displays
        self.log_text.clear()
        self.log_message("准备测试任务...")
        
        # Create worker and thread
        self.testing_worker = TestingWorker(
            model_path=self.model_path_edit.text(),
            test_dir=self.test_images_edit.text(),
            test_labels_dir=self.test_labels_edit.text(),
            output_dir=self.output_dir_edit.text(),
            dataset_format=self.dataset_format,
            conf_thresh=self.conf_thresh_spin.value(),
            iou_thresh=self.iou_thresh_spin.value(),
            img_size=self.img_size_spin.value(),
            save_results=self.save_results.isChecked()
        )
        
        self.testing_thread = QThread()
        self.testing_worker.moveToThread(self.testing_thread)
        
        # Connect signals
        self.testing_worker.progress_update.connect(self.update_progress)
        self.testing_worker.log_update.connect(self.log_message)
        self.testing_worker.metrics_update.connect(self.update_metrics)
        self.testing_worker.image_update.connect(self.update_image)
        self.testing_worker.testing_complete.connect(self.on_testing_complete)
        self.testing_worker.testing_error.connect(self.on_testing_error)
        self.testing_thread.started.connect(self.testing_worker.run)
        
        # Start testing
        self.testing_thread.start()
    
    def stop_testing(self):
        """Stop the testing process."""
        self.log_message("Stopping testing (please wait)...")
        
        # 更新状态
        self.status_label.setText("状态: 正在停止...")
        self.status_label.setStyleSheet("font-weight: bold; color: #f57900;")
        
        if self.testing_worker:
            self.testing_worker.stop()
    
    def on_testing_complete(self):
        """Handle testing completion."""
        self.is_testing = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        
        # 更新状态和进度
        self.status_label.setText("状态: 完成")
        self.progress_bar.setValue(100)
        
        self.log_message("Testing completed successfully!")
        
        QMessageBox.information(self, "Testing Complete", "Testing has been completed successfully.")
    
    def offer_retry_with_lower_thresholds(self):
        """提供使用较低阈值重试测试的选项。"""
        retry_msg = QMessageBox()
        retry_msg.setIcon(QMessageBox.Question)
        retry_msg.setWindowTitle("低性能检测")
        retry_msg.setText("检测到测试性能为0。这可能是由于置信度或IoU阈值设置过高。")
        retry_msg.setInformativeText("是否要以较低的阈值重新测试？")
        retry_msg.setDetailedText("将置信度阈值降低到0.1，IoU阈值降低到0.3，可能会检测到更多物体。")
        
        retry_button = retry_msg.addButton("重新测试", QMessageBox.ActionRole)
        cancel_button = retry_msg.addButton("取消", QMessageBox.RejectRole)
        
        retry_msg.exec_()
        
        if retry_msg.clickedButton() == retry_button:
            # 保存原始值以便在UI上显示
            original_conf = self.conf_thresh_spin.value()
            original_iou = self.iou_thresh_spin.value()
            
            # 设置新的低阈值
            self.conf_thresh_spin.setValue(0.1)
            self.iou_thresh_spin.setValue(0.3)
            
            # 记录阈值变化
            self.log_message(f"尝试使用较低阈值重新测试: 置信度 {original_conf} -> 0.1, IoU {original_iou} -> 0.3")
            
            # 重新开始测试
            self.start_testing()
    
    def on_testing_error(self, error_msg):
        """Handle testing error."""
        self.is_testing = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        
        # 更新状态
        self.status_label.setText("状态: 出错")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
        
        self.log_message(f"Error: {error_msg}")
        QMessageBox.critical(self, "Testing Error", f"An error occurred during testing:\n{error_msg}")
    
    def clean_up_thread(self):
        """Clean up thread and worker resources."""
        if self.testing_thread:
            self.testing_thread.quit()
            self.testing_thread.wait()
            self.testing_thread = None
            self.testing_worker = None
    
    def update_progress(self, progress):
        """Update progress bar."""
        # 更新进度条
        self.progress_bar.setValue(progress)
        
        # 更新状态标签
        self.status_label.setText(f"状态: 测试中 ({progress}%)")
        
        # 每10%记录一次进度到终端
        if progress % 10 == 0:
            self.log_message(f"测试进度: {progress}%")
    
    def log_message(self, message):
        """Add a message to the log display."""
        self.log_text.append(message)
        # Also print to stdout for terminal redirection
        print(f"[Testing] {message}")
        
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def update_metrics(self, metrics_text):
        """Update metrics display."""
        # 将指标作为日志添加到测试日志中
        if metrics_text:
            self.log_text.append("<pre>" + metrics_text + "</pre>")
            
            # 提取简要统计并显示在左侧信息面板
            try:
                # 尝试提取mAP和类别数量等关键指标
                lines = metrics_text.strip().split('\n')
                if len(lines) > 2:
                    # 通常最后一行包含总体指标
                    summary_line = [line for line in lines if 'all' in line.lower()]
                    if summary_line:
                        stats = summary_line[0].split()
                        if len(stats) >= 5:
                            map50 = stats[3]  # mAP@.5
                            map50_95 = stats[4]  # mAP@.5:.95
                            self.stats_label.setText(f"mAP@.5: {map50}\nmAP@.5:.95: {map50_95}")
                            return
                            
                # 如果无法提取详细指标，至少显示已更新
                self.stats_label.setText("已更新检测指标")
            except Exception as e:
                # 如果提取失败，不影响主程序
                print(f"提取指标时出错: {str(e)}")
                self.stats_label.setText("已更新检测指标")
    
    def update_image(self, image_path):
        """Update image display with detection result."""
        if os.path.exists(image_path):
            # 保存当前图像路径
            self.current_preview_image = image_path
            
            pixmap = QPixmap(image_path)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            label_size = self.image_display.size()
            scaled_pixmap = pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_display.setPixmap(scaled_pixmap)
        else:
            # Use an empty pixmap instead of text
            self.log_message(f"无法显示图像: {image_path} (文件不存在)")
            self.image_display.clear()
            self.image_display.setStyleSheet("background-color: #f0f0f0;")
            self.current_preview_image = None
    
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during testing."""
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.model_path_btn.setEnabled(enabled)
        self.test_images_btn.setEnabled(enabled)
        self.test_labels_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        self.conf_thresh_spin.setEnabled(enabled)
        self.iou_thresh_spin.setEnabled(enabled)
        self.img_size_spin.setEnabled(enabled)
        self.save_results.setEnabled(enabled)
        
        # 更新进度条状态
        if not enabled:
            self.progress_bar.setStyleSheet("QProgressBar { text-align: center; }")
        else:
            # 重置状态
            self.status_label.setText("状态: 就绪")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.progress_bar.setValue(0)
            self.stats_label.setText("等待测试开始...")
    
    def validate_inputs(self):
        """Validate user inputs before starting testing."""
        # Check if model path is set
        if not self.model_path_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a model file.")
            return False
        
        # Check if test directory is set
        if not self.test_images_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a test data directory.")
            return False
        
        # Check if output directory is set
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output directory.")
            return False
        
        return True
    
    def update_settings(self, settings):
        """Update tab settings based on settings from settings tab."""
        if 'default_conf_thresh' in settings:
            self.conf_thresh_spin.setValue(settings['default_conf_thresh'])
        
        if 'default_iou_thresh' in settings:
            self.iou_thresh_spin.setValue(settings['default_iou_thresh'])
        
        if 'default_img_size' in settings:
            self.img_size_spin.setValue(settings['default_img_size'])
        
        # Update default paths and set in UI if empty
        if 'default_test_dir' in settings:
            self.default_test_dir = settings['default_test_dir']
            if self.default_test_dir and not self.test_images_edit.text():
                self.test_images_edit.setText(self.default_test_dir)
        
        if 'default_output_dir' in settings:
            self.default_output_dir = settings['default_output_dir']
            if self.default_output_dir and not self.output_dir_edit.text():
                self.output_dir_edit.setText(self.default_output_dir)
            
        # Update default model path and set in UI if empty
        if 'default_test_model_path' in settings:
            self.default_model_path = settings['default_test_model_path']
            if self.default_model_path and not self.model_path_edit.text():
                self.model_path_edit.setText(self.default_model_path)
    
    def clear_terminal(self):
        """清除终端输出"""
        self.log_text.clear()
    
    def save_preview_image(self):
        """保存预览图片"""
        if not hasattr(self, 'current_preview_image') or not self.current_preview_image:
            QMessageBox.information(self, "保存失败", "没有可保存的预览图像")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存预览图像", "", "图像文件 (*.png *.jpg *.jpeg)"
        )
        
        if save_path:
            if not self.current_preview_image.endswith(('.png', '.jpg', '.jpeg')):
                QMessageBox.warning(self, "保存失败", "不支持的图像格式")
                return
                
            import shutil
            try:
                shutil.copy2(self.current_preview_image, save_path)
                QMessageBox.information(self, "保存成功", f"图像已保存至: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时出错: {str(e)}")
    
    def apply_theme_styles(self):
        """根据当前主题刷新所有控件样式，保证高对比度"""
        app = QApplication.instance()
        theme = app.property('theme') if app and app.property('theme') else 'tech'
        # 终端输出区
        if theme == 'light':
            self.log_text.setStyleSheet("background-color: #f7f7f7; color: #212529; border: 1px solid #ddd; border-radius: 4px; font-family: Consolas, monospace;")
        elif theme == 'dark':
            self.log_text.setStyleSheet("background-color: #1E1E1E; color: #DCE6F0; border: 1px solid #3F3F46; border-radius: 4px; font-family: Consolas, monospace;")
        else:  # tech
            self.log_text.setStyleSheet("background-color: #121A22; color: #DCE6F0; border: 1px solid #34465A; border-radius: 4px; font-family: Consolas, monospace;")
        # 主要标签和按钮
        label_color = {'light': '#212529', 'dark': '#DCE6F0', 'tech': '#DCE6F0'}[theme]
        for label in self.findChildren(QLabel):
            label.setStyleSheet(f"color: {label_color};")
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(btn.styleSheet() + f"color: {label_color};")
        # QGroupBox标题
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(f"QGroupBox {{ color: {label_color}; }}")
        # 进度条
        if theme == 'light':
            self.progress_bar.setStyleSheet("QProgressBar { background: #f7f7f7; color: #212529; border: 1px solid #ddd; border-radius: 4px; } QProgressBar::chunk { background: #4287f5; border-radius: 3px; }")
        elif theme == 'dark':
            self.progress_bar.setStyleSheet("QProgressBar { background: #252526; color: #DCE6F0; border: 1px solid #3F3F46; border-radius: 4px; } QProgressBar::chunk { background: #007ACC; border-radius: 3px; }")
        else:
            self.progress_bar.setStyleSheet("QProgressBar { background: #151E28; color: #DCE6F0; border: 1px solid #34465A; border-radius: 5px; } QProgressBar::chunk { background: #00BFFF; border-radius: 3px; }")
        # 图像显示区
        if hasattr(self, 'image_display'):
            if theme == 'light':
                self.image_display.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;")
            elif theme == 'dark':
                self.image_display.setStyleSheet("background-color: #1E1E1E; border: 1px solid #3F3F46; border-radius: 4px;")
            else:
                self.image_display.setStyleSheet("background-color: #121A22; border: 1px solid #34465A; border-radius: 4px;") 
    
    def validate_dataset(self):
        """Validate the test dataset structure and image-label matching."""
        # Get the directory paths
        test_images_dir = self.test_images_edit.text()
        test_labels_dir = self.test_labels_edit.text()
        
        # Check if paths are provided
        if not test_images_dir:
            QMessageBox.warning(self, "缺少路径", "请先选择测试图像目录")
            return
            
        # Validate the test dataset
        self.log_message("开始验证测试数据集...")
        test_results = validate_yolo_dataset(test_images_dir, test_labels_dir)
        
        # Log the test validation results
        self.log_message(f"测试数据集验证: {test_results['message']}")
        
        # Inspect dataset structure for more detailed information
        if test_images_dir:
            base_dir = os.path.dirname(os.path.dirname(test_images_dir))
            structure_report = inspect_dataset_structure(base_dir)
            self.log_message("\n数据集结构分析:\n" + structure_report)
        
        # Show summary result
        if test_results["success"]:
            QMessageBox.information(self, "验证成功", "测试数据集结构验证通过，可以开始测试。\n\n详细信息已添加到日志。")
        else:
            validation_message = (
                f"测试数据集验证发现以下问题:\n\n"
                f"测试集: 找到 {test_results['total_images']} 张图片, 匹配 {test_results['matched_labels']} 个标签\n\n"
                f"详细信息已添加到日志。\n\n"
                f"如果没有找到匹配的标签文件，测试将在推理模式下运行，不会评估精确度和召回率。"
            )
            
            QMessageBox.warning(self, "验证问题", validation_message) 