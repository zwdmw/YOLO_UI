import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QComboBox, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, 
                            QMessageBox, QProgressBar, QTextEdit, QRadioButton,
                            QButtonGroup, QFormLayout, QGridLayout, QSplitter,
                            QScrollArea, QTabWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication

from utils.inference_worker import InferenceWorker
from utils.theme_manager import ThemeManager

class InferenceTab(QWidget):
    """Tab for YOLO model inference on images or folders."""
    
    def __init__(self):
        super().__init__()
        self.is_inferencing = False
        self.inference_worker = None
        self.inference_thread = None
        
        # Default paths (will be updated from settings if available)
        self.default_output_dir = ""
        self.default_model_path = ""
        
        # Current preview image path
        self.current_preview_image = None
        self.original_pixmap = None
        
        # Set up UI
        self.setup_ui()
        self.apply_theme_styles()  # 初始化时应用主题样式
        
        # Install event filter to handle resize events
        self.installEventFilter(self)
        
    def setup_ui(self):
        """Create and arrange UI elements."""
        main_layout = QVBoxLayout(self)
        
        # Create a splitter with left panel for settings and right panel for results
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel - Settings
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(10, 10, 5, 10)  # Reduce horizontal margins
        
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
        
        # Inference mode section
        mode_group = QGroupBox("推理模式")
        mode_layout = QFormLayout()
        
        # Mode selection
        self.mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        
        self.image_radio = QRadioButton("图片模式")
        self.folder_radio = QRadioButton("文件夹模式")
        self.mode_group.addButton(self.image_radio)
        self.mode_group.addButton(self.folder_radio)
        self.mode_layout.addWidget(self.image_radio)
        self.mode_layout.addWidget(self.folder_radio)
        self.image_radio.setChecked(True)  # Default to image mode
        
        # Input selection
        self.input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_btn = QPushButton("浏览...")
        self.input_layout.addWidget(self.input_edit)
        self.input_layout.addWidget(self.input_btn)
        
        # Output directory
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        
        # Add widgets to form layout
        mode_layout.addRow("推理模式:", self.mode_layout)
        mode_layout.addRow("输入路径:", self.input_layout)
        mode_layout.addRow("输出目录:", self.output_dir_layout)
        
        # Save results option
        self.save_results = QCheckBox("保存检测结果")
        self.save_results.setChecked(True)
        mode_layout.addRow("", self.save_results)
        
        # Add view results button
        self.view_results_btn = QPushButton("查看检测结果")
        self.view_results_btn.setEnabled(False)  # Disabled until inference is completed
        mode_layout.addRow("", self.view_results_btn)
        
        mode_group.setLayout(mode_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始推理")
        self.start_btn.setMinimumHeight(40)
        self.stop_btn = QPushButton("停止推理")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        # Task info group
        info_group = QGroupBox("任务信息")
        info_layout = QVBoxLayout()
        
        # Status display
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        info_layout.addWidget(self.status_label)
        
        # Stats display
        self.stats_label = QLabel("等待推理开始...")
        info_layout.addWidget(self.stats_label)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        info_layout.addLayout(progress_layout)
        
        info_group.setLayout(info_layout)
        
        # Add sections to settings panel
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(mode_group)
        settings_layout.addLayout(control_layout)
        settings_layout.addWidget(info_group)
        settings_layout.addStretch(1)  # Add stretch to push everything up
        
        # Right Panel - Results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # Right side uses vertical splitter, top for result images, bottom for terminal output
        results_splitter = QSplitter(Qt.Vertical)
        
        # Upper part: Result image area
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        # Image header with title and save button
        image_header = QWidget()
        image_header_layout = QHBoxLayout(image_header)
        image_header_layout.setContentsMargins(0, 0, 0, 5)
        
        image_title = QLabel("检测结果预览")
        image_title.setFont(QFont("", 11, QFont.Bold))
        image_header_layout.addWidget(image_title)
        
        self.save_img_btn = QPushButton("保存图片")
        self.save_img_btn.setMaximumWidth(100)
        self.save_img_btn.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ccc; border-radius: 3px;")
        image_header_layout.addWidget(self.save_img_btn, alignment=Qt.AlignRight)
        
        image_layout.addWidget(image_header)
        
        # Image display area with overlay controls
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a custom label that handles resizing
        class ScalableImageLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.original_pixmap = None
                self.setAlignment(Qt.AlignCenter)
                self.setMinimumHeight(450)
                self.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ddd; border-radius: 4px;")
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
            def setPixmap(self, pixmap):
                self.original_pixmap = pixmap
                self.updatePixmap()
                
            def updatePixmap(self):
                if self.original_pixmap and not self.original_pixmap.isNull():
                    scaled_pixmap = self.original_pixmap.scaled(
                        self.width(), self.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    super().setPixmap(scaled_pixmap)
                    
            def resizeEvent(self, event):
                super().resizeEvent(event)
                self.updatePixmap()
        
        # Use the custom label for image display
        self.image_display = ScalableImageLabel()
        
        # Put image directly in the layout without scroll area
        image_container_layout.addWidget(self.image_display, 1)
        
        # Image browser controls - moved to image area and improved visibility
        self.img_browser_controls = QWidget()
        self.img_browser_controls.setStyleSheet("""
            QWidget {
                background-color: rgba(240, 240, 240, 0.85);
                border-top: 1px solid #ccc;
            }
            QPushButton {
                padding: 5px 10px;
                background-color: #e0e0e0;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QLabel {
                font-weight: bold;
                color: #333;
            }
        """)
        img_browser_layout = QHBoxLayout(self.img_browser_controls)
        img_browser_layout.setContentsMargins(10, 10, 10, 10)
        
        self.prev_img_btn = QPushButton("« 上一张")
        self.prev_img_btn.setCursor(Qt.PointingHandCursor)
        
        self.next_img_btn = QPushButton("下一张 »")
        self.next_img_btn.setCursor(Qt.PointingHandCursor)
        
        self.img_counter_label = QLabel("0/0")
        self.img_counter_label.setAlignment(Qt.AlignCenter)
        self.img_counter_label.setMinimumWidth(80)
        
        self.close_browser_btn = QPushButton("关闭浏览器")
        self.close_browser_btn.setCursor(Qt.PointingHandCursor)
        
        img_browser_layout.addWidget(self.prev_img_btn)
        img_browser_layout.addWidget(self.img_counter_label)
        img_browser_layout.addWidget(self.next_img_btn)
        img_browser_layout.addWidget(self.close_browser_btn)
        
        self.img_browser_controls.setVisible(False)
        image_container_layout.addWidget(self.img_browser_controls)
        
        # Add the image container to the image layout
        image_layout.addWidget(image_container)
        
        # Lower part: Terminal output
        terminal_widget = QWidget()
        terminal_layout = QVBoxLayout(terminal_widget)
        terminal_layout.setContentsMargins(5, 5, 5, 5)
        
        # Terminal header
        terminal_header = QWidget()
        terminal_header_layout = QHBoxLayout(terminal_header)
        terminal_header_layout.setContentsMargins(0, 0, 0, 5)
        
        terminal_title = QLabel("终端输出")
        terminal_title.setFont(QFont("", 11, QFont.Bold))
        terminal_header_layout.addWidget(terminal_title)
        
        clear_btn = QPushButton("清除")
        clear_btn.setMaximumWidth(80)
        clear_btn.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ccc; border-radius: 3px;")
        clear_btn.clicked.connect(self.clear_terminal)
        terminal_header_layout.addWidget(clear_btn, alignment=Qt.AlignRight)
        
        terminal_layout.addWidget(terminal_header)
        
        # Terminal text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.document().setDefaultStyleSheet("pre {margin: 0; font-family: monospace;}")
        self.log_text.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ddd; border-radius: 4px; font-family: Consolas, monospace;")
        terminal_layout.addWidget(self.log_text)
        
        # Add components to results splitter
        results_splitter.addWidget(image_widget)
        results_splitter.addWidget(terminal_widget)
        results_splitter.setSizes([600, 200])  # Adjust size ratio to favor image area
        
        # Set stretch factors for the results splitter
        results_splitter.setStretchFactor(0, 3)  # Image area gets more stretch
        results_splitter.setStretchFactor(1, 1)  # Terminal area gets less stretch
        
        # Add splitter to results panel
        results_layout.addWidget(results_splitter)
        
        # Add panels to main splitter
        main_splitter.addWidget(settings_panel)
        main_splitter.addWidget(results_panel)
        
        # Add main splitter to layout
        main_layout.addWidget(main_splitter)
        
        # Set initial size proportions - make results panel larger
        main_splitter.setSizes([200, 800])  # Adjust size ratio to give more space to results
        
        # Set stretch factors to maintain proportions during resize
        main_splitter.setStretchFactor(0, 0)  # Settings panel doesn't stretch
        main_splitter.setStretchFactor(1, 1)  # Results panel gets all the stretch
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to their respective slots."""
        # Mode selection
        self.image_radio.toggled.connect(self.update_mode)
        self.folder_radio.toggled.connect(self.update_mode)
        
        # Path selection
        self.model_path_btn.clicked.connect(self.select_model_path)
        self.input_btn.clicked.connect(self.select_input_path)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        
        # Control buttons
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn.clicked.connect(self.stop_inference)
        self.save_img_btn.clicked.connect(self.save_preview_image)
        
        # Image browser controls
        self.view_results_btn.clicked.connect(self.open_image_browser)
        self.prev_img_btn.clicked.connect(self.show_prev_image)
        self.next_img_btn.clicked.connect(self.show_next_image)
        self.close_browser_btn.clicked.connect(self.close_image_browser)
        
        # 监听主题切换信号
        from ui.main_window import MainWindow
        main_window = self.parentWidget()
        while main_window and not isinstance(main_window, MainWindow):
            main_window = main_window.parentWidget()
        if main_window and hasattr(main_window, 'settings_tab'):
            main_window.settings_tab.theme_changed.connect(lambda _: self.apply_theme_styles())
    
    def update_mode(self, checked):
        """Update input selection button text based on selected mode."""
        if self.image_radio.isChecked():
            self.input_btn.setText("选择图片...")
            self.input_edit.setPlaceholderText("选择要检测的图片")
        elif self.folder_radio.isChecked():
            self.input_btn.setText("选择文件夹...")
            self.input_edit.setPlaceholderText("选择包含图片的文件夹")
        
        # Clear the input field when changing modes
        self.input_edit.clear()
    
    def select_model_path(self):
        """Open dialog to select model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.pth *.weights);;所有文件 (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def select_input_path(self):
        """Open dialog to select input (image or folder)."""
        if self.image_radio.isChecked():
            # Image mode - select multiple files
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
            )
            if file_paths:
                # Join multiple paths with semicolons for display
                self.input_edit.setText(";".join(file_paths))
                if len(file_paths) > 1:
                    self.log_message(f"已选择 {len(file_paths)} 张图片")
        else:
            # Folder mode - select directory
            dir_path = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
            if dir_path:
                self.input_edit.setText(dir_path)
    
    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def start_inference(self):
        """Validate inputs and start inference in a separate thread."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable UI elements during inference
        self.set_ui_enabled(False)
        self.is_inferencing = True
        
        # Reset status and progress
        self.status_label.setText("状态: 准备中")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        self.progress_bar.setValue(0)
        self.stats_label.setText("正在加载模型...")
        
        # Clear displays
        self.log_text.clear()
        self.log_message("准备推理任务...")
        
        # Create worker and thread
        self.inference_worker = InferenceWorker(
            model_path=self.model_path_edit.text(),
            input_path=self.input_edit.text(),
            output_dir=self.output_dir_edit.text(),
            is_folder_mode=self.folder_radio.isChecked(),
            conf_thresh=self.conf_thresh_spin.value(),
            iou_thresh=self.iou_thresh_spin.value(),
            img_size=self.img_size_spin.value(),
            save_results=self.save_results.isChecked()
        )
        
        self.inference_thread = QThread()
        self.inference_worker.moveToThread(self.inference_thread)
        
        # Connect signals
        self.inference_worker.progress_update.connect(self.update_progress)
        self.inference_worker.log_update.connect(self.log_message)
        self.inference_worker.stats_update.connect(self.update_stats)
        self.inference_worker.image_update.connect(self.update_image)
        self.inference_worker.inference_complete.connect(self.on_inference_complete)
        self.inference_worker.inference_error.connect(self.on_inference_error)
        self.inference_thread.started.connect(self.inference_worker.run)
        
        # Start inference
        self.inference_thread.start()
    
    def stop_inference(self):
        """Stop the inference process."""
        self.log_message("正在停止推理(请稍候)...")
        
        # Update status
        self.status_label.setText("状态: 正在停止...")
        self.status_label.setStyleSheet("font-weight: bold; color: #f57900;")
        
        if self.inference_worker:
            self.inference_worker.stop()
    
    def on_inference_complete(self):
        """Handle completion of inference process."""
        self.log_message("推理完成")
        
        # Update UI
        self.is_inferencing = False
        self.set_ui_enabled(True)
        self.status_label.setText("状态: 已完成")
        self.progress_bar.setValue(100)
        
        # Enable the view results button if results should be saved
        if self.save_results.isChecked():
            self.view_results_btn.setEnabled(True)
            
            # Automatically open the image browser if there are results
            results_dir = os.path.join(self.output_dir_edit.text(), 'inference_results')
            if os.path.exists(results_dir):
                image_files = [f for f in os.listdir(results_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
                if image_files and len(image_files) > 0:
                    # Ask if user wants to view results immediately
                    reply = QMessageBox.question(
                        self, '查看结果', 
                        f"推理已完成，是否立即查看检测结果？\n(共 {len(image_files)} 张图片)",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        self.open_image_browser()
        
        # Clean up thread
        self.clean_up_thread()
    
    def on_inference_error(self, error_msg):
        """Handle inference error."""
        self.is_inferencing = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        
        # Update status
        self.status_label.setText("状态: 出错")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
        
        self.log_message(f"错误: {error_msg}")
        QMessageBox.critical(self, "推理错误", f"推理过程中发生错误:\n{error_msg}")
    
    def clean_up_thread(self):
        """Clean up thread and worker resources."""
        if self.inference_thread:
            self.inference_thread.quit()
            self.inference_thread.wait()
            self.inference_thread = None
            self.inference_worker = None
    
    def update_progress(self, progress):
        """Update progress bar."""
        # Update progress bar
        self.progress_bar.setValue(progress)
        
        # Update status label
        self.status_label.setText(f"状态: 推理中 ({progress}%)")
        
        # Log progress at 10% intervals
        if progress % 10 == 0:
            self.log_message(f"推理进度: {progress}%")
    
    def log_message(self, message):
        """Add a message to the log display."""
        self.log_text.append(message)
        # Also print to stdout for terminal redirection
        print(f"[推理] {message}")
        
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def update_stats(self, stats_text):
        """Update statistics display."""
        self.stats_label.setText(stats_text)
    
    def update_image(self, image_path):
        """Update image display with detection result."""
        if os.path.exists(image_path):
            # Save current image path
            self.current_preview_image = image_path
            
            # Load the image and set it to our scalable label
            pixmap = QPixmap(image_path)
            self.image_display.setPixmap(pixmap)
                
            # If this is being called from the image browser, ensure controls are visible
            if hasattr(self, 'image_files') and self.image_files:
                self.img_browser_controls.setVisible(True)
        else:
            self.log_message(f"无法显示图像: {image_path} (文件不存在)")
            self.image_display.clear()
            self.image_display.setStyleSheet("background-color: #f0f0f0;")
            self.current_preview_image = None
    
    def resizeEvent(self, event):
        """Main widget resize event - pass to parent"""
        super().resizeEvent(event)
        
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during inference."""
        # Model settings
        self.model_path_btn.setEnabled(enabled)
        self.conf_thresh_spin.setEnabled(enabled)
        self.iou_thresh_spin.setEnabled(enabled)
        self.img_size_spin.setEnabled(enabled)
        
        # Mode settings
        self.image_radio.setEnabled(enabled)
        self.folder_radio.setEnabled(enabled)
        self.input_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        self.save_results.setEnabled(enabled)
        
        # Start/stop buttons
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        
        # Don't disable view results button when starting/stopping inference
        # It should remain enabled if results are available
        
        # Update progress bar status
        if not enabled:
            self.progress_bar.setStyleSheet("QProgressBar { text-align: center; }")
        else:
            # Reset status
            self.status_label.setText("状态: 就绪")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.progress_bar.setValue(0)
            self.stats_label.setText("等待推理开始...")
    
    def validate_inputs(self):
        """Validate user inputs before starting inference."""
        # Check if model path is set
        if not self.model_path_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择模型文件。")
            return False
        
        # Check if input path is set
        if not self.input_edit.text():
            if self.image_radio.isChecked():
                QMessageBox.warning(self, "缺少输入", "请选择要检测的图片。")
            else:
                QMessageBox.warning(self, "缺少输入", "请选择包含图片的文件夹。")
            return False
        
        # Check if output directory is set
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择输出目录。")
            return False
        
        # In folder mode, check if directory exists and contains images
        if self.folder_radio.isChecked():
            input_dir = self.input_edit.text()
            if not os.path.isdir(input_dir):
                QMessageBox.warning(self, "无效输入", "所选路径不是有效的文件夹。")
                return False
            
            # Check if directory contains any images
            has_images = False
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        has_images = True
                        break
                if has_images:
                    break
            
            if not has_images:
                QMessageBox.warning(self, "无效输入", "所选文件夹不包含任何图片文件。")
                return False
        
        # In image mode, check if files exist and are images
        if self.image_radio.isChecked():
            image_paths = self.input_edit.text().split(';')
            invalid_paths = []
            
            for image_path in image_paths:
                # Check if file exists
                if not os.path.isfile(image_path):
                    invalid_paths.append(f"{image_path} (不是有效的文件)")
                    continue
                
                # Check if file is an image
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                if not any(image_path.lower().endswith(ext) for ext in image_extensions):
                    invalid_paths.append(f"{image_path} (不是支持的图片格式)")
            
            if invalid_paths:
                error_msg = "以下文件无效:\n" + "\n".join(invalid_paths)
                QMessageBox.warning(self, "无效输入", error_msg)
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
        
        # Update default output directory
        if 'default_output_dir' in settings:
            self.default_output_dir = settings['default_output_dir']
            if self.default_output_dir and not self.output_dir_edit.text():
                self.output_dir_edit.setText(self.default_output_dir)
            
        # Update default model path
        if 'default_test_model_path' in settings:
            self.default_model_path = settings['default_test_model_path']
            if self.default_model_path and not self.model_path_edit.text():
                self.model_path_edit.setText(self.default_model_path)
    
    def clear_terminal(self):
        """Clear terminal output"""
        self.log_text.clear()
    
    def save_preview_image(self):
        """Save preview image"""
        if not hasattr(self, 'current_preview_image') or not self.current_preview_image:
            QMessageBox.information(self, "保存失败", "没有可保存的预览图像")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存预览图像", "", "图像文件 (*.png *.jpg *.jpeg)"
        )
        
        if save_path:
            # Make sure file has an extension
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path += '.jpg'
                
            import shutil
            try:
                shutil.copy2(self.current_preview_image, save_path)
                QMessageBox.information(self, "保存成功", f"图像已保存至: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时出错: {str(e)}")
    
    # Add image browser functionality
    def open_image_browser(self):
        """Open the image browser to view detection results."""
        # Get the results directory
        results_dir = os.path.join(self.output_dir_edit.text(), 'inference_results')
        
        if not os.path.exists(results_dir):
            QMessageBox.warning(self, "无法打开图片浏览器", 
                               f"结果目录不存在: {results_dir}\n请先运行推理。")
            return
        
        # Get all image files in the directory
        self.image_files = [f for f in os.listdir(results_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        if not self.image_files:
            QMessageBox.information(self, "没有检测结果", 
                                  "结果目录中没有找到图片文件。")
            return
        
        # Initialize browser
        self.current_img_index = 0
        
        # Show image browser controls
        self.img_browser_controls.setVisible(True)
        
        # Update counter display
        self.img_counter_label.setText(f"1/{len(self.image_files)}")
        
        # Update status to indicate browsing mode
        self.status_label.setText("状态: 浏览检测结果")
        self.status_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        
        # Log the action
        self.log_message(f"打开图片浏览器, 共找到 {len(self.image_files)} 张图片")
        
        # Show first image
        self.display_current_image()
        
        # Make sure UI updates are processed
        QCoreApplication.processEvents()
        
    def display_current_image(self):
        """Display the current image in the browser."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
            
        # Get the results directory
        results_dir = os.path.join(self.output_dir_edit.text(), 'inference_results')
        
        # Get current image path
        img_path = os.path.join(results_dir, self.image_files[self.current_img_index])
        
        # Update counter
        self.img_counter_label.setText(f"{self.current_img_index + 1}/{len(self.image_files)}")
        
        # Load and display image
        self.update_image(img_path)
        
        # Update current preview image
        self.current_preview_image = img_path
        
    def show_next_image(self):
        """Show the next image in the browser."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
            
        self.current_img_index = (self.current_img_index + 1) % len(self.image_files)
        self.display_current_image()
        
    def show_prev_image(self):
        """Show the previous image in the browser."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
            
        self.current_img_index = (self.current_img_index - 1) % len(self.image_files)
        self.display_current_image()
        
    def close_image_browser(self):
        """Close the image browser."""
        self.img_browser_controls.setVisible(False)
        
        # Reset status
        self.status_label.setText("状态: 就绪")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        
        # Clear the displayed image
        self.image_display.clear()
        self.image_display.setText("无预览")
        self.current_preview_image = None 

    def eventFilter(self, obj, event):
        """Event filter - no longer needed for image scaling"""
        return super().eventFilter(obj, event) 

    def apply_theme_styles(self):
        """根据当前主题刷新所有控件样式，保证高对比度"""
        app = QApplication.instance()
        theme = app.property('theme') if app and app.property('theme') else 'tech'
        # 终端输出区
        if hasattr(self, 'log_text'):
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
        if hasattr(self, 'progress_bar'):
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