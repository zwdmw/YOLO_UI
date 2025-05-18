# Add buttons layout
buttons_layout = QHBoxLayout()

# Select model button
self.select_model_btn = QPushButton("选择模型")
self.select_model_btn.clicked.connect(self.select_model)
buttons_layout.addWidget(self.select_model_btn)

# Select test data button
self.select_test_data_btn = QPushButton("选择测试数据")
self.select_test_data_btn.clicked.connect(self.select_test_data)
buttons_layout.addWidget(self.select_test_data_btn)

# Select output directory button
self.select_output_dir_btn = QPushButton("选择输出目录")
self.select_output_dir_btn.clicked.connect(self.select_output_dir)
buttons_layout.addWidget(self.select_output_dir_btn)

# Start testing button
self.start_test_btn = QPushButton("开始测试")
self.start_test_btn.clicked.connect(self.start_testing)
buttons_layout.addWidget(self.start_test_btn)

# Stop testing button
self.stop_test_btn = QPushButton("停止测试")
self.stop_test_btn.clicked.connect(self.stop_testing)
self.stop_test_btn.setEnabled(False)
buttons_layout.addWidget(self.stop_test_btn)

# 添加生成伪标注按钮（默认禁用，只有在测试完成且未找到标注时才启用）
self.generate_labels_btn = QPushButton("生成伪标注")
self.generate_labels_btn.clicked.connect(self.generate_pseudo_labels)
self.generate_labels_btn.setEnabled(False)
self.generate_labels_btn.setToolTip("从模型预测生成伪标注用于评估\n注意：这些标注仅用于调试目的")
buttons_layout.addWidget(self.generate_labels_btn)

main_layout.addLayout(buttons_layout)

def handle_testing_complete(self):
    """Handle testing completion."""
    self.progress_bar.setValue(100)
    self.start_test_btn.setEnabled(True)
    self.stop_test_btn.setEnabled(False)
    
    # 检查是否可以启用伪标注按钮（我们可以从日志中查找特定的错误消息）
    log_text = self.log_text.toPlainText().lower()
    if "未找到有效的标签目录" in log_text or "no valid label directory" in log_text:
        self.generate_labels_btn.setEnabled(True)
    else:
        self.generate_labels_btn.setEnabled(False)
        
    QMessageBox.information(self, "测试完成", "模型测试已完成!")

def generate_pseudo_labels(self):
    """从模型预测生成伪标注"""
    if not self.model_path or not self.test_data_path:
        QMessageBox.warning(self, "参数缺失", "请先选择模型和测试数据")
        return
        
    # 确认用户操作
    reply = QMessageBox.question(
        self,
        "生成伪标注",
        "此操作将使用模型检测结果作为'伪标注'生成标签文件。\n\n"
        "请注意:\n"
        "1. 这些标注基于模型自身的预测，仅用于调试目的\n"
        "2. 标注将保存在测试数据对应的labels目录中\n"
        "3. 此过程可能需要一些时间\n\n"
        "确认继续?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        # 提交到测试线程
        self.create_testing_thread(
            model_path=self.model_path,
            test_dir=self.test_data_path,
            output_dir=self.output_dir,
            dataset_format=self.dataset_format_combo.currentText(),
            conf_thresh=self.conf_thresh_slider.value() / 100,
            iou_thresh=self.iou_thresh_slider.value() / 100,
            img_size=int(self.img_size_combo.currentText()),
            save_results=self.save_results_check.isChecked(),
            generate_pseudo_labels=True  # 指示生成伪标注的模式
        )
        
        # 更新UI
        self.start_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(True)
        self.generate_labels_btn.setEnabled(False)
        self.append_to_log("开始生成伪标注...")

def create_testing_thread(self, model_path, test_dir, output_dir, dataset_format,
                          conf_thresh, iou_thresh, img_size, save_results,
                          generate_pseudo_labels=False):
    """Create and start the testing thread."""
    # Create worker instance
    self.testing_worker = TestingWorker(
        model_path=model_path,
        test_dir=test_dir,
        output_dir=output_dir,
        dataset_format=dataset_format,
        conf_thresh=conf_thresh, 
        iou_thresh=iou_thresh,
        img_size=img_size,
        save_results=save_results
    )
    
    # Connect signals
    self.testing_worker.progress_update.connect(self.update_progress)
    self.testing_worker.log_update.connect(self.append_to_log)
    self.testing_worker.metrics_update.connect(self.update_metrics)
    self.testing_worker.image_update.connect(self.update_image)
    self.testing_worker.testing_complete.connect(self.handle_testing_complete)
    self.testing_worker.testing_error.connect(self.handle_testing_error)
    
    # Create thread and move worker to it
    self.testing_thread = QThread()
    self.testing_worker.moveToThread(self.testing_thread)
    
    # Connect thread start/finish signals
    self.testing_thread.started.connect(
        self.testing_worker.generate_pseudo_labels if generate_pseudo_labels 
        else self.testing_worker.run
    )
    self.testing_thread.finished.connect(self.testing_thread.deleteLater)
    
    # Start thread
    self.testing_thread.start() 