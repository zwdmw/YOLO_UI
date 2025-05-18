import os
import sys
import time
import traceback
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import cv2
import torch
import io

class TestingWorker(QObject):
    """Worker for running YOLO model testing in a separate thread."""
    
    # Signals
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(str)
    image_update = pyqtSignal(str)
    testing_complete = pyqtSignal()
    testing_error = pyqtSignal(str)
    
    def __init__(self, model_path, test_dir, output_dir, dataset_format="COCO", 
                 conf_thresh=0.25, iou_thresh=0.45, img_size=640, save_results=True, test_labels_dir=None):
        """Initialize the testing worker with parameters."""
        super().__init__()
        self.model_path = model_path
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.dataset_format = dataset_format
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.save_results = save_results
        self.should_stop = False
        self.test_labels_dir = test_labels_dir
        
    def stop(self):
        """Signal the worker to stop processing."""
        self.should_stop = True
        self.log_update.emit("Received stop signal")
    
    def run(self):
        """Execute the testing process."""
        try:
            self.log_update.emit(f"Starting testing with model: {os.path.basename(self.model_path)}")
            self.log_update.emit(f"Test directory: {self.test_dir}")
            if self.test_labels_dir:
                self.log_update.emit(f"Test labels directory: {self.test_labels_dir}")
            self.log_update.emit(f"Dataset format: {self.dataset_format}")
            self.log_update.emit(f"Confidence threshold: {self.conf_thresh}")
            self.log_update.emit(f"IoU threshold: {self.iou_thresh}")
            self.log_update.emit(f"Image size: {self.img_size}")
            
            # Make sure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load YOLO model
            self.load_model()
            
            if self.should_stop:
                self.log_update.emit("Testing stopped")
                return
            
            # First try to create a YOLO-compatible dataset for proper evaluation
            if self.prepare_yolo_compatible_dataset():
                self.log_update.emit("使用标准验证方式评估模型性能...")
                self.run_standard_validation()
            else:
                try:    
                    # Process test data based on format
                    if self.dataset_format == "COCO":
                        self.test_coco_dataset()
                    elif self.dataset_format == "VOC":
                        self.test_voc_dataset()
                    elif self.dataset_format == "YOLO":
                        self.test_yolo_dataset()
                    else:
                        raise ValueError(f"Unsupported dataset format: {self.dataset_format}")
                except Exception as e:
                    # If validation fails (possibly due to missing annotations), fall back to inference
                    self.log_update.emit(f"Validation error: {str(e)}")
                    self.log_update.emit("Falling back to inference-only mode (no ground truth comparison)")
                    self.run_inference_on_images()
                
            if not self.should_stop:
                self.testing_complete.emit()
                
        except Exception as e:
            error_msg = f"Error during testing: {str(e)}\n{traceback.format_exc()}"
            self.testing_error.emit(error_msg)
    
    def load_model(self):
        """Load the YOLO model."""
        self.log_update.emit("Loading YOLO model...")
        
        try:
            # Use Ultralytics YOLO
            import ultralytics
            from ultralytics import YOLO
            
            # Log the Ultralytics version
            self.log_update.emit(f"Using Ultralytics version: {ultralytics.__version__}")
            
            # Check if this is a yolo12 model
            is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
            
            # Load the model with appropriate parameters
            if is_yolo12:
                self.log_update.emit("检测到yolo12模型，使用兼容模式加载")
                self.model = YOLO(self.model_path, task='detect')
            else:
                # Regular YOLO model loading
                self.model = YOLO(self.model_path)
            
            # Log model information
            self.log_update.emit(f"Model loaded: {self.model_path}")
            self.log_update.emit(f"Model task: {self.model.task}")
            self.log_update.emit(f"Model loaded successfully")
            
            # Check model attributes for compatibility
            self.ultralytics_version = "v8+" if hasattr(self.model, 'predict') else "v5"
            self.log_update.emit(f"Detected API compatibility: Ultralytics {self.ultralytics_version}")
            
            # Try to load custom class names from classes.txt or dataset.yaml
            custom_names = self._get_custom_class_names()
            if custom_names and hasattr(self.model, 'names'):
                self.log_update.emit(f"使用自定义类名覆盖模型默认类名")
                if len(custom_names) < len(self.model.names):
                    self.log_update.emit(f"警告：自定义类名数量({len(custom_names)})小于模型类别数量({len(self.model.names)})")
                    custom_names.extend([f"class{i}" for i in range(len(custom_names), len(self.model.names))])
                self._apply_custom_names(custom_names)
            
        except ImportError:
            self.log_update.emit("Ultralytics package not found. Falling back to torch hub...")
            try:
                # Fallback to torch hub
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                self.model.conf = self.conf_thresh
                self.model.iou = self.iou_thresh
                self.model.classes = None  # all classes
                self.model.max_det = 300  # maximum number of detections
                self.ultralytics_version = "torch_hub"
                self.log_update.emit("Model loaded from torch hub")
                
                # Try to load custom class names from classes.txt or dataset.yaml
                custom_names = self._get_custom_class_names()
                if custom_names and hasattr(self.model, 'names'):
                    self.log_update.emit(f"使用自定义类名覆盖模型默认类名")
                    # If the custom class names list is shorter than the model's classes, extend it
                    if len(custom_names) < len(self.model.names):
                        self.log_update.emit(f"警告：自定义类名数量({len(custom_names)})小于模型类别数量({len(self.model.names)})")
                        custom_names.extend([f"class{i}" for i in range(len(custom_names), len(self.model.names))])
                    self.model.names = custom_names
            except Exception as e:
                raise ValueError(f"Failed to load model: {str(e)}")
    
    def _get_custom_class_names(self):
        """从classes.txt或dataset.yaml(data.yaml)文件中获取自定义类名"""
        class_names = []
        
        # 获取可能的目录路径
        model_dir = os.path.dirname(self.model_path)
        
        # 可能的yaml文件路径
        possible_yaml_paths = [
            os.path.join(model_dir, "dataset.yaml"),
            os.path.join(model_dir, "data.yaml"),
            os.path.join(self.test_dir, "dataset.yaml"),
            os.path.join(self.test_dir, "data.yaml"),
            os.path.join(os.path.dirname(self.test_dir), "dataset.yaml"),
            os.path.join(os.path.dirname(self.test_dir), "data.yaml"),
            os.path.join(self.output_dir, "dataset.yaml"),
            os.path.join(self.output_dir, "data.yaml"),
        ]
        
        # 首先尝试从yaml文件获取
        for yaml_path in possible_yaml_paths:
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    if 'names' in data:
                        self.log_update.emit(f"从YAML文件加载类名: {yaml_path}")
                        # 处理不同格式的names（列表或字典）
                        if isinstance(data['names'], list):
                            return data['names']
                        elif isinstance(data['names'], dict):
                            # 确保按索引排序
                            sorted_names = [None] * len(data['names'])
                            for idx, name in data['names'].items():
                                sorted_names[int(idx)] = name
                            return sorted_names
                except Exception as e:
                    self.log_update.emit(f"从YAML读取类名失败: {str(e)}")
        
        # 然后尝试从classes.txt获取
        possible_class_files = [
            os.path.join(model_dir, "classes.txt"),
            os.path.join(self.test_dir, "classes.txt"),
            os.path.join(os.path.dirname(self.test_dir), "classes.txt"),
            os.path.join(self.output_dir, "classes.txt"),
        ]
        
        # 如果有标签目录，也检查它
        if self.test_labels_dir and os.path.exists(self.test_labels_dir):
            possible_class_files.append(os.path.join(self.test_labels_dir, "classes.txt"))
        
        for class_file in possible_class_files:
            if os.path.exists(class_file):
                try:
                    with open(class_file, 'r', encoding='utf-8') as f:
                        class_names = [line.strip() for line in f.readlines() if line.strip()]
                    
                    if class_names:
                        self.log_update.emit(f"从classes.txt加载类名: {class_file}")
                        return class_names
                except Exception as e:
                    self.log_update.emit(f"从classes.txt读取类名失败: {str(e)}")
        
        return []
    
    def test_coco_dataset(self):
        """Test on a COCO format dataset."""
        self.log_update.emit("Testing on COCO format dataset...")
        
        # Check if annotations exist
        annotations_file = os.path.join(self.test_dir, 'annotations', 'instances_default.json')
        
        if not os.path.exists(annotations_file):
            self.log_update.emit(f"COCO annotations file not found: {annotations_file}")
            self.log_update.emit("Will run inference-only mode instead")
            self.run_inference_on_images()
            return
        
        # Run validation on the model
        try:
            self.log_update.emit("Starting validation...")
            
            # Create a basic validation params dict
            val_params = {
                'data': self.test_dir,
                'conf': self.conf_thresh,
                'iou': self.iou_thresh,
                'imgsz': self.img_size,
                'save_json': True,
                'save_txt': self.save_results,
                'save_conf': self.save_results,
                'project': self.output_dir,
                'name': 'test_results',
                'verbose': True
            }
            
            # Try to use the correct API based on version
            try:
                # 使用新版API参数名称
                results = self.model.val(**val_params)
            except TypeError as e:
                # 检查错误是否与参数有关
                self.log_update.emit(f"API参数错误: {str(e)}, 尝试向下兼容参数")
                # 尝试转换imgsz为size (YOLOv5向下兼容)
                if 'imgsz' in val_params:
                    val_params['size'] = val_params.pop('imgsz')
                    self.log_update.emit("将'imgsz'参数转换为'size'参数")
                try:
                    results = self.model.val(**val_params)
                except Exception as e2:
                    self.log_update.emit(f"尝试兼容参数后仍然失败: {str(e2)}")
                    raise
            
        
            # Display sample images
            self.display_sample_images()
            
        except Exception as e:
            self.log_update.emit(f"Validation error: {str(e)}")
            self.log_update.emit("Falling back to inference-only mode")
            self.run_inference_on_images()
    
    def test_voc_dataset(self):
        """Test on a VOC format dataset."""
        self.log_update.emit("Testing on VOC format dataset...")
        
        # Check if annotations exist
        annotations_dir = os.path.join(self.test_dir, 'Annotations')
        
        if not os.path.exists(annotations_dir) or not os.listdir(annotations_dir):
            self.log_update.emit(f"VOC annotations directory not found or empty: {annotations_dir}")
            self.log_update.emit("Will run inference-only mode instead")
            self.run_inference_on_images()
            return
        
        # Run validation on the model
        try:
            self.log_update.emit("Starting validation...")
            
            # Create a basic validation params dict
            val_params = {
                'data': self.test_dir,
                'conf': self.conf_thresh,
                'iou': self.iou_thresh,
                'imgsz': self.img_size,
                'save_txt': self.save_results,
                'save_conf': self.save_results,
                'project': self.output_dir,
                'name': 'test_results',
                'verbose': True
            }
            
            # Try to use the correct API based on version
            results = self.model.val(**val_params)
            
            # Display metrics
            # self.display_metrics(results)
            
            # Display sample images
            self.display_sample_images()
            
        except Exception as e:
            self.log_update.emit(f"Validation error: {str(e)}")
            self.log_update.emit("Falling back to inference-only mode")
            self.run_inference_on_images()
    
    def test_yolo_dataset(self):
        """Test on a YOLO format dataset."""
        self.log_update.emit("Testing on YOLO format dataset...")
        
        # Look for dataset.yaml in the test directory
        yaml_file = os.path.join(self.test_dir, 'dataset.yaml')
        if not os.path.exists(yaml_file):
            # Try data.yaml as an alternative
            yaml_file = os.path.join(self.test_dir, 'data.yaml')
            if not os.path.exists(yaml_file):
                yaml_file = None
                self.log_update.emit("Warning: dataset.yaml/data.yaml not found in test directory")
        
        # 优先使用test_labels_dir
        labels_dir = self.test_labels_dir if self.test_labels_dir else os.path.join(self.test_dir, 'labels')
        if not os.path.exists(labels_dir) or not os.listdir(labels_dir):
            self.log_update.emit(f"YOLO labels directory not found or empty: {labels_dir}")
            self.log_update.emit("Will run inference-only mode instead")
            self.run_inference_on_images()
            return
        
        # Run validation on the model
        try:
            self.log_update.emit("Starting validation...")
            
            # Create a basic validation params dict
            val_params = {
                'data': yaml_file or self.test_dir,
                'conf': self.conf_thresh,
                'iou': self.iou_thresh,
                'imgsz': self.img_size,
                'save_txt': self.save_results,
                'save_conf': self.save_results,
                'project': self.output_dir,
                'name': 'test_results',
                'verbose': True
            }
            
            # Try to use the correct API based on version
            results = self.model.val(**val_params)
            
            # Display metrics
            # self.display_metrics(results)
            
            # Display sample images
            self.display_sample_images()
            
        except Exception as e:
            self.log_update.emit(f"Validation error: {str(e)}")
            self.log_update.emit("Falling back to inference-only mode")
            self.run_inference_on_images()
    
    def run_inference_on_images(self):
        """Run inference on images if there are no ground truth annotations."""
        self.log_update.emit("运行推理中...")
        
        # Find all images in the test directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        detection_counts = {}
        total_detections = 0
        
        # Search for images based on dataset format
        if self.dataset_format == "COCO":
            # COCO format typically has images in an 'images' directory
            images_dir = os.path.join(self.test_dir, 'images')
            if os.path.exists(images_dir):
                self.log_update.emit(f"Searching for images in COCO images directory: {images_dir}")
                self._find_images(images_dir, image_files, image_extensions)
        elif self.dataset_format == "VOC":
            # VOC format typically has images in a 'JPEGImages' directory
            images_dir = os.path.join(self.test_dir, 'JPEGImages')
            if os.path.exists(images_dir):
                self.log_update.emit(f"Searching for images in VOC JPEGImages directory: {images_dir}")
                self._find_images(images_dir, image_files, image_extensions)
        
        # If no images found with specific format paths, search the entire directory
        if not image_files:
            self.log_update.emit(f"Searching for images in the entire test directory: {self.test_dir}")
            self._find_images(self.test_dir, image_files, image_extensions)
        
        if not image_files:
            self.log_update.emit("No image files found in the test directory")
            self.metrics_update.emit("未找到任何图片文件，请检查测试数据目录")
            return
        
        self.log_update.emit(f"Found {len(image_files)} images")
        
        # Create output directory
        results_dir = os.path.join(self.output_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Process images
        for i, img_path in enumerate(image_files):
            if self.should_stop:
                self.log_update.emit("Testing stopped")
                return
            
            # Update progress
            progress = int((i + 1) / len(image_files) * 100)
            self.progress_update.emit(progress)
            
            # Log current file
            self.log_update.emit(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            # Run inference
            try:
                # Check if this is a yolo12 model
                is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
                
                # Run inference with appropriate API
                if is_yolo12:
                    # yolo12 API requires specific parameters
                    self.log_update.emit("使用yolo12 API进行推理")
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                else:
                    # 首先尝试新版的Ultralytics v8 API
                    try:
                        results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                    except TypeError as e:
                        # 记录错误并尝试兼容参数
                        self.log_update.emit(f"API参数错误: {str(e)}, 尝试向下兼容参数")
                        # 使用旧版的参数名称 (size)
                        results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            except Exception as e:
                self.log_update.emit(f"推理失败: {str(e)}")
                continue
            
            # Count detections for summary
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"class{cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"class{cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                # Debug log for detection count
                if "(no detections)" not in str(results):
                    self.log_update.emit(f"检测到目标：在图片 {os.path.basename(img_path)} 中")
            except Exception as e:
                self.log_update.emit(f"统计检测结果时出错: {str(e)}")
            
            # Save results
            if self.save_results:
                if hasattr(results, 'save'):
                    # Ultralytics v8 API
                    results.save(save_dir=results_dir)
                else:
                    # Manual save for older API
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Draw results on the image
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"class{int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        output_path = os.path.join(results_dir, os.path.basename(img_path))
                        cv2.imwrite(output_path, img)
            
            # Display sample image
            try:
                if i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1:  # Show ~10 samples
                    # First try Ultralytics v8 API
                    if hasattr(results, 'plot'):
                        result_img = results[0].plot()
                        result_path = os.path.join(results_dir, f"sample_{i}_{os.path.basename(img_path)}")
                        cv2.imwrite(result_path, result_img)
                        self.image_update.emit(result_path)
                    # Then try Ultralytics v5 API
                    elif hasattr(results, 'render'):
                        results.render()  # Updates results.imgs with boxes
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            result_path = os.path.join(results_dir, f"sample_{i}_{os.path.basename(img_path)}")
                            cv2.imwrite(result_path, results.imgs[0])
                            self.image_update.emit(result_path)
                    # Fallback to the saved image
                    else:
                        output_path = os.path.join(results_dir, os.path.basename(img_path))
                        if os.path.exists(output_path):
                            self.image_update.emit(output_path)
                        else:
                            self.log_update.emit(f"Could not find output image: {os.path.basename(output_path)}")
            except Exception as e:
                self.log_update.emit(f"Error displaying image: {str(e)}")
                # Try to use the saved image as fallback
                output_path = os.path.join(results_dir, os.path.basename(img_path))
                if os.path.exists(output_path):
                    self.image_update.emit(output_path)
        
        # Update metrics with basic inference stats and detection counts
        self.log_update.emit("生成检测统计报告...")
        
        inference_metrics = ""
        if detection_counts:
            inference_metrics += "检测统计:\n"
            inference_metrics += f"{'-'*30}\n"
            inference_metrics += f"{'类别':15s} {'检测数量':8s} {'占比'}\n"
            inference_metrics += f"{'-'*30}\n"
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                inference_metrics += f"{cls_name[:15]:15s} {count:<8d} {percentage:.1f}%\n"
        
        # 确保性能指标明显显示出来
        self.log_update.emit(f"总图片数: {len(image_files)} | 检测目标数: {total_detections}")
        
        # 记录类别统计
        if detection_counts:
            self.log_update.emit("类别分布:")
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                self.log_update.emit(f"  {cls_name}: {count}个 ({percentage:.1f}%)")
        
        # 更新UI上的指标显示
        self.metrics_update.emit(inference_metrics)
    
    def _find_images(self, directory, image_files, extensions):
        """Find all images with given extensions in a directory and its subdirectories."""
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_files.append(os.path.join(root, file))
    
    def display_metrics(self, results):
        """Display testing metrics."""
        try:
            self.log_update.emit("处理并显示检测指标...")
            
            metrics_text = "检测精度指标:\n"
            metrics_text += f"{'类别':10s} {'精确率':10s} {'召回率':10s} {'mAP50':10s} {'mAP50-95':10s}\n"
            metrics_text += f"{'-'*55}\n"
            
            # 特别处理YOLOv8格式的results对象
            if hasattr(results, 'maps') and isinstance(results.maps, (list, tuple, np.ndarray)) and len(results.maps) > 0:
                try:
                    self.log_update.emit("处理YOLOv8格式的性能指标...")
                    
                    # 检查maps属性，这里包含各个类别的mAP值
                    # 提取总体指标
                    if hasattr(results, 'box') and hasattr(results.box, 'map50'):
                        map50 = float(results.box.map50)
                        map = float(results.box.map)
                        
                        # 如果有precision和recall属性
                        precision = float(results.box.precision) if hasattr(results.box, 'precision') else 0
                        recall = float(results.box.recall) if hasattr(results.box, 'recall') else 0
                        
                        metrics_text += f"{'所有类别':10s} {precision:.4f}    {recall:.4f}    {map50:.4f}    {map:.4f}\n"
                        self.log_update.emit(f"总体性能: mAP50={map50:.4f}, mAP50-95={map:.4f}")
                        
                        # 尝试提取各类别性能
                        if isinstance(results.maps, (list, tuple, np.ndarray)) and len(results.maps) > 0:
                            metrics_text += ""
                            
                            
                    self.metrics_update.emit(metrics_text)
                    return
                except Exception as e:
                    self.log_update.emit(f"处理YOLOv8 maps格式指标时出错: {str(e)}")
                    import traceback
                    self.log_update.emit(traceback.format_exc())
            
            # 如果有results_dict属性
            if hasattr(results, 'results_dict') and results.results_dict:
                try:
                    self.log_update.emit("从results_dict中提取性能指标...")
                    results_dict = results.results_dict
                    
                    # 提取总体指标
                    if 'metrics/mAP50(B)' in results_dict and 'metrics/mAP50-95(B)' in results_dict:
                        map50 = float(results_dict.get('metrics/mAP50(B)', 0))
                        map = float(results_dict.get('metrics/mAP50-95(B)', 0))
                        precision = float(results_dict.get('metrics/precision(B)', 0))
                        recall = float(results_dict.get('metrics/recall(B)', 0))
                        
                        metrics_text += f"{'所有类别':10s} {precision:.4f}    {recall:.4f}    {map50:.4f}    {map:.4f}\n"
                        self.log_update.emit(f"总体性能: mAP50={map50:.4f}, mAP50-95={map:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}")
                    
                    self.metrics_update.emit(metrics_text)
                    return
                except Exception as e:
                    self.log_update.emit(f"处理results_dict指标时出错: {str(e)}")
            
            # 尝试其他可能的结果格式
            if hasattr(results, 'ap_class_index') and hasattr(results, 'maps') and isinstance(results.maps, (list, tuple, np.ndarray)) and len(results.maps) > 0:
                try:
                    # YOLOv5格式
                    if hasattr(results, 'mp') and hasattr(results, 'mr'):
                        mp = float(results.mp)  # 平均精确率
                        mr = float(results.mr)  # 平均召回率
                        
                        # 安全地处理maps数组
                        if isinstance(results.maps, (list, tuple, np.ndarray)) and len(results.maps) > 0:
                            map50 = float(results.maps[0])  # mAP@0.5 - 取maps中的第一个元素
                            map = float(sum(results.maps) / len(results.maps))  # mAP@0.5:0.95 - 所有IoU阈值下的平均
                        else:
                            map50 = 0
                            map = 0
                        
                        metrics_text += f"{'所有类别':10s} {mp:.4f}    {mr:.4f}    {map50:.4f}    {map:.4f}\n"
                        self.log_update.emit(f"总体性能: mAP50={map50:.4f}, mAP50-95={map:.4f}, 精确率={mp:.4f}, 召回率={mr:.4f}")
                        
                        # 各类别详细性能
                        if isinstance(results.ap_class_index, (list, tuple, np.ndarray)) and len(results.ap_class_index) > 0:
                            metrics_text += ""
                    
                    self.metrics_update.emit(metrics_text)
                    return
                except Exception as e:
                    self.log_update.emit(f"处理V5格式指标时出错: {str(e)}")
                    import traceback
                    self.log_update.emit(traceback.format_exc())
            
            # 如果没有标准格式，尝试从属性中直接提取
            try:
                self.log_update.emit("尝试从结果对象属性直接提取数据...")
                
                # 尝试提取性能数据
                extracted_map50 = 0
                extracted_map = 0
                extracted_precision = 0
                extracted_recall = 0
                extracted = False
                
                # 尝试从box属性提取数据
                if hasattr(results, 'box'):
                    box = results.box
                    if hasattr(box, 'map50'):
                        extracted_map50 = float(box.map50)
                        extracted = True
                    if hasattr(box, 'map'):
                        extracted_map = float(box.map)
                        extracted = True
                    if hasattr(box, 'precision'):
                        extracted_precision = float(box.precision)
                    if hasattr(box, 'recall'):
                        extracted_recall = float(box.recall)
                
                # 尝试从curves_results中提取数据
                elif hasattr(results, 'curves_results') and results.curves_results:
                    curves = results.curves_results
                    for curve in curves:
                        if 'mAP@0.5' in curve['name']:
                            extracted_map50 = float(curve['value'])
                            extracted = True
                        elif 'mAP@0.5:0.95' in curve['name']:
                            extracted_map = float(curve['value'])
                            extracted = True
                        elif 'precision' in curve['name'].lower():
                            extracted_precision = float(curve['value'])
                        elif 'recall' in curve['name'].lower():
                            extracted_recall = float(curve['value'])
                
                # 如果提取到了数据
                if extracted:
                    metrics_text += f"{'所有类别':10s} {extracted_precision:.4f}    {extracted_recall:.4f}    {extracted_map50:.4f}    {extracted_map:.4f}\n"
                    self.log_update.emit(f"提取到性能指标: mAP50={extracted_map50:.4f}, mAP50-95={extracted_map:.4f}")
                    self.metrics_update.emit(metrics_text)
                    return
                
                # 如果仍然没有找到，使用原始代码尝试提取
                if hasattr(results, 'ap') and hasattr(results, 'p') and hasattr(results, 'r'):
                    # 安全处理NumPy数组
                    ap = results.ap
                    ap50 = results.ap[0] if isinstance(ap, (list, tuple, np.ndarray)) and len(ap) > 0 else 0  # 假设第一个元素是mAP@0.5
                    
                    # 计算平均值时确保是数值
                    if isinstance(ap, (list, tuple, np.ndarray)):
                        ap_mean = float(np.mean(ap))
                    elif isinstance(ap, (int, float)):
                        ap_mean = float(ap)
                    else:
                        ap_mean = 0
                    
                    # 同样安全处理p和r
                    p = results.p
                    if isinstance(p, (list, tuple, np.ndarray)):
                        mp = float(np.mean(p))
                    elif isinstance(p, (int, float)):
                        mp = float(p)
                    else:
                        mp = 0
                    
                    r = results.r
                    if isinstance(r, (list, tuple, np.ndarray)):
                        mr = float(np.mean(r))
                    elif isinstance(r, (int, float)):
                        mr = float(r)
                    else:
                        mr = 0
                    
                    metrics_text += f"{'所有类别':10s} {mp:.4f}    {mr:.4f}    {ap50:.4f}    {ap_mean:.4f}\n"
                    self.log_update.emit(f"提取到性能指标: mAP50={ap50:.4f}, mAP50-95={ap_mean:.4f}")
                    self.metrics_update.emit(metrics_text)
                    return
                
                # 从fitness属性提取
                if hasattr(results, 'fitness') and float(results.fitness) > 0:
                    metrics_text += f"模型性能分数: {float(results.fitness):.4f}\n\n"
                    metrics_text += "未能提取详细精确率/召回率数据，但模型验证成功完成。\n"
                    metrics_text += "请查看日志获取更多信息。"
                    self.log_update.emit(f"提取到模型性能分数: {float(results.fitness):.4f}")
                    self.metrics_update.emit(metrics_text)
                    return
                
                # 最后的兜底方案
                metrics_text = "无法提取标准性能指标，但模型验证已完成\n\n"
                metrics_text += "请检查以下可能的原因:\n"
                metrics_text += "1. 数据集是否包含有效的标注文件\n"
                metrics_text += "2. 标注格式是否与模型类别匹配\n"
                metrics_text += "3. 检测阈值设置是否合适\n\n"
                try:
                    attrs = [attr for attr in dir(results) if not attr.startswith('_') and not callable(getattr(results, attr))]
                    metrics_text += "可用属性: " + ", ".join(attrs)
                except:
                    metrics_text += "无法提取结果对象属性"
                self.metrics_update.emit(metrics_text)
            except Exception as e:
                self.log_update.emit(f"尝试提取指标时出错: {str(e)}")
                import traceback
                self.log_update.emit(traceback.format_exc())
                metrics_text = "处理性能指标时出错，请查看日志获取详细信息"
                self.metrics_update.emit(metrics_text)
        except Exception as e:
            self.log_update.emit(f"显示性能指标时出错: {str(e)}")
            import traceback
            self.log_update.emit(traceback.format_exc())
            # 确保无论如何都会更新UI
            self.metrics_update.emit("性能指标处理错误，请查看日志了解详情")
    
    def display_sample_images(self):
        """Display sample images from test results."""
        try:
            self.log_update.emit("查找并显示检测结果样例...")
            
            # 查找结果目录
            results_dir = os.path.join(self.output_dir, 'test_results')
            if not os.path.exists(results_dir):
                self.log_update.emit(f"未找到结果目录: {results_dir}")
                return
            
            # 查找图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            result_images = []
            
            for root, _, files in os.walk(results_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        result_images.append(os.path.join(root, file))
            
            if not result_images:
                self.log_update.emit("未找到任何结果图像")
                
                # 尝试查找原始测试图像并显示
                original_images = []
                self._find_images(self.test_dir, original_images, image_extensions)
                
                if original_images:
                    sample_image = np.random.choice(original_images)
                    self.log_update.emit(f"未找到结果图像，显示原始样例图片: {os.path.basename(sample_image)}")
                    
                    # 尝试以原始图像进行一次推理并显示结果
                    try:
                        # 创建一个临时保存路径
                        temp_result_dir = os.path.join(self.output_dir, 'temp_preview')
                        os.makedirs(temp_result_dir, exist_ok=True)
                        
                        # 运行一次推理
                        self.log_update.emit("创建预览图像...")
                        result_filename = os.path.join(temp_result_dir, f"preview_{os.path.basename(sample_image)}")
                        
                        # 使用模型进行推理
                        try:
                            # 先尝试新的API (imgsz)
                            results = self.model(sample_image, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                        except TypeError:
                            # 再尝试旧的API (size)
                            results = self.model(sample_image, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                        
                        # 保存图像
                        if hasattr(results, 'plot'):
                            # YOLOv8 API
                            result_img = results[0].plot()
                            cv2.imwrite(result_filename, result_img)
                        elif hasattr(results, 'render'):
                            # YOLOv5 API
                            results.render()
                            if hasattr(results, 'imgs') and len(results.imgs) > 0:
                                cv2.imwrite(result_filename, results.imgs[0])
                        else:
                            # 手动绘制
                            img = cv2.imread(sample_image)
                            # [手动绘制逻辑]
                            cv2.imwrite(result_filename, img)
                        
                        # 更新UI
                        if os.path.exists(result_filename):
                            self.image_update.emit(result_filename)
                            self.log_update.emit(f"已创建预览图像: {os.path.basename(result_filename)}")
                        else:
                            self.log_update.emit("未能创建预览图像")
                            self.image_update.emit(sample_image)  # 回退到显示原始图像
                            
                    except Exception as e:
                        self.log_update.emit(f"创建预览图像时出错: {str(e)}")
                        self.image_update.emit(sample_image)  # 回退到显示原始图像
                return
            
            # 选择一个图像显示
            # 优先选择带有"sample"或"result"的文件，这些通常是带有检测框的图片
            priority_images = [img for img in result_images if 'sample' in img.lower() or 'result' in img.lower()]
            
            if priority_images:
                sample_image = np.random.choice(priority_images)
            else:
                sample_image = np.random.choice(result_images)
                
            self.log_update.emit(f"显示检测结果图片: {os.path.basename(sample_image)}")
            self.image_update.emit(sample_image)
            
        except Exception as e:
            self.log_update.emit(f"显示样例图片时出错: {str(e)}")
    
    def prepare_yolo_compatible_dataset(self):
        """
        检查测试数据目录是否包含YOLO格式的标注，或将其它格式转换为YOLO格式。
        返回是否成功找到或创建了YOLO格式的数据集。
        """
        self.log_update.emit("检查测试数据集格式...")
        
        # 查找images目录或者图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images_dir = os.path.join(self.test_dir, 'images')
        image_files = []
        has_images_dir = os.path.exists(images_dir) and os.listdir(images_dir)
        
        if has_images_dir:
            self.log_update.emit(f"找到images目录: {images_dir}")
            self._find_images(images_dir, image_files, image_extensions)
        else:
            self.log_update.emit("未找到标准images目录，在测试目录中搜索图片")
            self._find_images(self.test_dir, image_files, image_extensions)
        
        if not image_files:
            self.log_update.emit("未找到任何图片文件")
            return False
        
        self.log_update.emit(f"找到 {len(image_files)} 张图片")
        
        # 检查可能的标注目录，优先test_labels_dir
        possible_label_dirs = []
        if self.test_labels_dir:
            possible_label_dirs.append(self.test_labels_dir)
        possible_label_dirs += [
            os.path.join(self.test_dir, 'labels'),
            os.path.join(self.test_dir, 'Annotations'),
            os.path.join(self.test_dir, 'annotations'),
            os.path.join(self.test_dir, 'label'),
            os.path.join(self.test_dir, 'ann'),
            os.path.join(self.test_dir, 'gt')
        ]
        
        # 查找同级目录中的标注文件
        found_labels = False
        label_dir_path = None
        
        for dir_path in possible_label_dirs:
            if os.path.exists(dir_path) and os.listdir(dir_path):
                self.log_update.emit(f"找到可能的标注目录: {dir_path}")
                label_dir_path = dir_path
                
                # 检查此目录中是否有与图片匹配的标注文件
                matching_labels_count = 0
                for img_file in image_files:
                    img_basename = os.path.splitext(os.path.basename(img_file))[0]
                    label_file = os.path.join(dir_path, f"{img_basename}.txt")
                    if os.path.exists(label_file):
                        matching_labels_count += 1
                
                if matching_labels_count > 0:
                    self.log_update.emit(f"在标注目录中找到 {matching_labels_count} 个与图片匹配的标注文件")
                    found_labels = True
                    break
                else:
                    self.log_update.emit(f"在标注目录 {dir_path} 中未找到与图片匹配的标注文件")
                
        # 如果在标准目录中没有找到匹配的标注，尝试在images同级查找
        if not found_labels:
            self.log_update.emit("在images同级目录中查找标注文件...")
            # 获取测试目录中所有可能的标注文件
            all_files = []
            for root, _, files in os.walk(self.test_dir):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    # 常见的标注文件扩展名
                    if file_ext in ['.txt', '.xml', '.json', '.yml', '.yaml']:
                        all_files.append(os.path.join(root, file))
            
            if all_files:
                # 查找是否有与图片同名的txt文件 (YOLO格式标注)
                image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in image_files]
                matching_txt_files = []
                
                for file_path in all_files:
                    if file_path.lower().endswith('.txt'):
                        basename = os.path.splitext(os.path.basename(file_path))[0]
                        if basename in image_basenames:
                            matching_txt_files.append(file_path)
                
                if matching_txt_files:
                    self.log_update.emit(f"找到 {len(matching_txt_files)} 个与图片匹配的标注文件")
                    found_labels = True
                    
                    # 创建YOLO格式的目录结构
                    yolo_dataset_dir = os.path.join(self.output_dir, 'yolo_dataset')
                    yolo_images_dir = os.path.join(yolo_dataset_dir, 'images')
                    yolo_labels_dir = os.path.join(yolo_dataset_dir, 'labels')
                    
                    os.makedirs(yolo_images_dir, exist_ok=True)
                    os.makedirs(yolo_labels_dir, exist_ok=True)
                    
                    self.log_update.emit(f"创建YOLO格式数据集目录: {yolo_dataset_dir}")
                    
                    # 复制图片和标注到YOLO目录结构
                    copied_count = 0
                    for txt_file in matching_txt_files:
                        txt_basename = os.path.splitext(os.path.basename(txt_file))[0]
                        # 找到对应的图片
                        for img_file in image_files:
                            img_basename = os.path.splitext(os.path.basename(img_file))[0]
                            if img_basename == txt_basename:
                                # 复制标注文件
                                dst_txt = os.path.join(yolo_labels_dir, os.path.basename(txt_file))
                                with open(txt_file, 'r') as src, open(dst_txt, 'w') as dst:
                                    dst.write(src.read())
                                
                                # 复制图片文件
                                img_ext = os.path.splitext(img_file)[1]
                                dst_img = os.path.join(yolo_images_dir, txt_basename + img_ext)
                                import shutil
                                shutil.copy2(img_file, dst_img)
                                copied_count += 1
                    
                    self.log_update.emit(f"成功复制了 {copied_count} 对图片和标注文件")
                    
                    # 创建data.yaml文件
                    if hasattr(self.model, 'names'):
                        yaml_path = os.path.join(yolo_dataset_dir, 'data.yaml')
                        with open(yaml_path, 'w') as f:
                            f.write(f"path: {yolo_dataset_dir}\n")
                            f.write("train: images/train\n")
                            f.write("val: images\n")
                            f.write("test: images\n\n")
                            f.write(f"nc: {len(self.model.names)}\n")
                            f.write(f"names: {list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names}\n")
                    
                    # 更新测试目录为新创建的YOLO数据集
                    self.test_dir = yolo_dataset_dir
                    self.log_update.emit(f"已准备YOLO格式数据集，包含 {copied_count} 张图片和对应标注")
                    return True
        
        # 如果已经找到了标注目录，但是没有创建新的数据集，检查是否需要设置为YOLO标准目录结构
        if found_labels and label_dir_path:
            # 检查是否需要创建标准YOLO目录结构
            if not (has_images_dir and label_dir_path == os.path.join(self.test_dir, 'labels')):
                # 创建YOLO格式的目录结构
                yolo_dataset_dir = os.path.join(self.output_dir, 'yolo_dataset')
                yolo_images_dir = os.path.join(yolo_dataset_dir, 'images')
                yolo_labels_dir = os.path.join(yolo_dataset_dir, 'labels')
                
                os.makedirs(yolo_images_dir, exist_ok=True)
                os.makedirs(yolo_labels_dir, exist_ok=True)
                
                self.log_update.emit(f"创建YOLO格式数据集目录: {yolo_dataset_dir}")
                
                # 复制图片和标注到YOLO目录结构
                copied_count = 0
                for img_file in image_files:
                    img_basename = os.path.splitext(os.path.basename(img_file))[0]
                    label_file = os.path.join(label_dir_path, f"{img_basename}.txt")
                    
                    if os.path.exists(label_file):
                        # 复制标注文件
                        dst_txt = os.path.join(yolo_labels_dir, f"{img_basename}.txt")
                        with open(label_file, 'r') as src, open(dst_txt, 'w') as dst:
                            dst.write(src.read())
                        
                        # 复制图片文件
                        img_ext = os.path.splitext(img_file)[1]
                        dst_img = os.path.join(yolo_images_dir, img_basename + img_ext)
                        import shutil
                        shutil.copy2(img_file, dst_img)
                        copied_count += 1
                
                self.log_update.emit(f"成功复制了 {copied_count} 对图片和标注文件")
                
                # 创建data.yaml文件
                if hasattr(self.model, 'names'):
                    yaml_path = os.path.join(yolo_dataset_dir, 'data.yaml')
                    with open(yaml_path, 'w') as f:
                        f.write(f"path: {yolo_dataset_dir}\n")
                        f.write("train: images/train\n")
                        f.write("val: images\n")
                        f.write("test: images\n\n")
                        f.write(f"nc: {len(self.model.names)}\n")
                        f.write(f"names: {list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names}\n")
                
                # 更新测试目录为新创建的YOLO数据集
                self.test_dir = yolo_dataset_dir
                self.log_update.emit(f"已准备YOLO格式数据集，包含 {copied_count} 张图片和对应标注")
                return True
            else:
                # 已经是标准YOLO目录结构
                self.log_update.emit("找到标准YOLO格式数据集")
                return True
        
        # 尝试查找YOLO格式的配置文件
        yaml_file = None
        for yaml_name in ['data.yaml', 'dataset.yaml']:
            possible_yaml = os.path.join(self.test_dir, yaml_name)
            if os.path.exists(possible_yaml):
                yaml_file = possible_yaml
                self.log_update.emit(f"找到数据集配置文件: {yaml_name}")
                break
        
        # 如果有标准的YOLO目录结构，返回True
        labels_dir = os.path.join(self.test_dir, 'labels')
        has_labels = os.path.exists(labels_dir) and os.listdir(labels_dir)
        
        if has_images_dir and has_labels:
            if yaml_file:
                self.log_update.emit("找到完整的YOLO格式数据集，可以进行标准验证")
            else:
                self.log_update.emit("找到图片和标注目录，但缺少data.yaml配置文件")
                # 自动创建简单的data.yaml
                if hasattr(self.model, 'names'):
                    yaml_path = os.path.join(self.test_dir, 'data.yaml')
                    with open(yaml_path, 'w') as f:
                        f.write(f"path: {self.test_dir}\n")
                        f.write("train: images/train\n")
                        f.write("val: images\n")
                        f.write("test: images\n\n")
                        f.write(f"nc: {len(self.model.names)}\n")
                        f.write(f"names: {list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names}\n")
                    self.log_update.emit(f"已创建基本的data.yaml配置文件: {yaml_path}")
            return True
        
        # 检查COCO格式
        if self.dataset_format == "COCO":
            annotations_file = os.path.join(self.test_dir, 'annotations', 'instances_default.json')
            if os.path.exists(annotations_file):
                self.log_update.emit("找到COCO格式标注，将使用COCO评估方式")
                return False  # 使用原生COCO评估
        
        # 检查VOC格式
        if self.dataset_format == "VOC":
            annotations_dir = os.path.join(self.test_dir, 'Annotations')
            if os.path.exists(annotations_dir) and os.listdir(annotations_dir):
                self.log_update.emit("找到VOC格式标注，将使用VOC评估方式")
                return False  # 使用原生VOC评估
        
        # 如果只有图片，没有标注
        if image_files:
            self.log_update.emit(f"在测试目录下找到 {len(image_files)} 张图片，但无法找到与之匹配的标注数据")
            self.log_update.emit("将使用推理模式，无法计算精确率、召回率和mAP指标")
            return False
        
        self.log_update.emit("无法找到适合测试的数据集格式")
        return False

    def run_standard_validation(self):
        """使用YOLO标准验证流程进行评估"""
        self.log_update.emit("正在运行标准验证流程...")
        try:
            # 查找数据集配置文件
            yaml_file = None
            for yaml_name in ['data.yaml', 'dataset.yaml']:
                possible_yaml = os.path.join(self.test_dir, yaml_name)
                if os.path.exists(possible_yaml):
                    yaml_file = possible_yaml
                    break
            
            if not yaml_file:
                self.log_update.emit("未找到数据集配置文件，尝试使用目录路径")
            
            # 创建结果目录
            results_dir = os.path.join(self.output_dir, 'test_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 运行验证
            self.log_update.emit("执行模型验证...")
            
            # 创建临时日志拦截器来捕获YOLO输出
            original_stdout = sys.stdout
            log_capture = io.StringIO()
            sys.stdout = log_capture
            
            try:
                val_params = {
                    'data': yaml_file or self.test_dir,
                    'conf': self.conf_thresh,
                    'iou': self.iou_thresh,
                    'imgsz': self.img_size,
                    'save_txt': self.save_results,
                    'save_conf': self.save_results,
                    'project': self.output_dir,
                    'name': 'test_results',
                    'verbose': True,
                    'save_json': True,
                    'save': True,  # 保存检测图像结果
                }
                
                # 尝试不同的API参数
                try:
                    results = self.model.val(**val_params)
                except TypeError as e:
                    self.log_update.emit(f"API参数错误: {str(e)}, 尝试向下兼容参数")
                    # 尝试转换imgsz为size (YOLOv5向下兼容)
                    if 'imgsz' in val_params:
                        val_params['size'] = val_params.pop('imgsz')
                    # 移除可能不兼容的参数
                    for param in ['plots', 'save_json']:
                        if param in val_params:
                            val_params.pop(param)
                    results = self.model.val(**val_params)
            finally:
                # 恢复标准输出并处理捕获的日志
                sys.stdout = original_stdout
                log_output = log_capture.getvalue()
                
                # 去除ANSI颜色代码
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_log = ansi_escape.sub('', log_output)
                
                # 从输出中提取性能表格
                performance_table = []
                inside_table = False
                val_start = False
                
                for line in clean_log.split('\n'):
                    # 记录从val开始的输出
                    if 'val: Scanning' in line:
                        val_start = True
                    
                    if val_start:
                        # 记录包含性能指标的行
                        if 'all' in line and ('mAP50' in line or 'P' in line):
                            inside_table = True
                            performance_table.append(line.strip())
                        elif inside_table and len(line.strip()) > 0 and line[0].isspace():
                            # 继续收集表格行（缩进的行）
                            performance_table.append(line.strip())
                        elif inside_table and 'Results saved' in line:
                            # 表格结束
                            performance_table.append(line.strip())
                            inside_table = False
                            break
                
                # 输出处理后的表格
                self.log_update.emit("检测性能结果:")
                for line in performance_table:
                    self.log_update.emit(line)
                
                # 构建UI显示的格式化性能表格
                if performance_table:
                    # 解析性能表格为UI友好格式
                    metrics_text = "检测精度指标:\n"
                    metrics_text += f"{'类别':12s} {'图片数':8s} {'目标数':8s} {'精确率':8s} {'召回率':8s} {'mAP50':8s} {'mAP50-95':8s}\n"
                    metrics_text += f"{'-'*70}\n"
                    
                    # 处理每一行
                    for line in performance_table:
                        # 跳过非数据行
                        if '%' in line or 'Results saved' in line:
                            continue
                        
                        # 清理行内容，替换多个空格为单个空格
                        clean_line = re.sub(r'\s+', ' ', line.strip())
                        parts = clean_line.split()
                        
                        if len(parts) >= 7:  # 确保有足够的列
                            class_name = parts[0]
                            images = parts[1]
                            instances = parts[2]
                            precision = parts[3]
                            recall = parts[4]
                            map50 = parts[5]
                            map50_95 = parts[6]
                            
                            # 添加到表格
                            metrics_text += f"{class_name:12s} {images:8s} {instances:8s} {precision:8s} {recall:8s} {map50:8s} {map50_95:8s}\n"
                    
                    # 更新UI显示
                    self.metrics_update.emit(metrics_text)
                else:
                    # 如果没有找到性能表格，使用标准方法显示
                    print('')
                
                # 保留重要的信息行，用于日志显示
                important_lines = []
                for line in clean_log.split('\n'):
                    # 保留有用的信息，过滤冗余
                    if ('val:' in line or 'mAP' in line or 'Precision' in line or 
                        'Recall' in line) and '0/0' not in line and '%' not in line:
                        important_lines.append(line.strip())
                
                # 输出处理后的日志
                for line in important_lines:
                    if line not in performance_table:  # 避免重复输出
                        self.log_update.emit(line)
            
            # 尝试寻找验证生成的图像
            self.log_update.emit("查找验证结果图像...")
            
            # 从model.val()结果中获取save_dir
            save_dir = None
            if hasattr(results, 'save_dir') and results.save_dir:
                save_dir = results.save_dir
                self.log_update.emit(f"找到结果保存目录: {save_dir}")
            else:
                save_dir = results_dir
                self.log_update.emit(f"使用默认结果目录: {save_dir}")
            
            # 创建一个预测图像并显示
            self.log_update.emit("创建预测图像...")
            
            # 测试单张图像以生成预览
            images_dir = os.path.join(self.test_dir, 'images')
            if not os.path.exists(images_dir):
                self.log_update.emit("未找到标准images目录，搜索整个测试目录")
                images_dir = self.test_dir
            
            # 查找所有图像
            image_files = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            self._find_images(images_dir, image_files, image_extensions)
            
            if not image_files:
                self.log_update.emit("找不到任何图像文件")
                return
            
            # 首先检查是否已经有验证结果图像
            result_images = []
            for root, _, files in os.walk(save_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        if ('val_batch' in file.lower() or 'pred' in file.lower() or 
                            'labels' in file.lower()):
                            result_images.append(os.path.join(root, file))
            
            if result_images:
                # 使用已有的结果图像
                sample_image = result_images[0]  # 使用第一张图片
                self.log_update.emit(f"使用验证结果图像: {os.path.basename(sample_image)}")
                self.image_update.emit(sample_image)
                return
                
            # 如果没有找到结果图像，创建新的预览
            # 创建预览目录
            preview_dir = os.path.join(self.output_dir, 'preview')
            os.makedirs(preview_dir, exist_ok=True)
            
            # 随机选择一张图片生成预览
            sample_img_path = np.random.choice(image_files)
            self.log_update.emit(f"使用图像生成预览: {os.path.basename(sample_img_path)}")
            
            preview_path = os.path.join(preview_dir, f"preview_{os.path.basename(sample_img_path)}")
            
            # 运行推理并保存
            try:
                # 尝试使用新API
                try:
                    pred_results = self.model(sample_img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                except TypeError:
                    # 尝试使用旧API
                    pred_results = self.model(sample_img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                
                # 保存预测结果图像
                if hasattr(pred_results[0], 'plot'):
                    # YOLOv8 API
                    self.log_update.emit("使用YOLOv8 API绘制预测结果")
                    result_img = pred_results[0].plot()
                    cv2.imwrite(preview_path, result_img)
                elif hasattr(pred_results, 'render'):
                    # YOLOv5 API
                    self.log_update.emit("使用YOLOv5 API绘制预测结果")
                    pred_results.render()
                    if hasattr(pred_results, 'imgs') and len(pred_results.imgs) > 0:
                        cv2.imwrite(preview_path, pred_results.imgs[0])
                else:
                    # 手动绘制
                    self.log_update.emit("使用手动方式绘制预测结果")
                    img = cv2.imread(sample_img_path)
                    # 在图像上绘制检测框
                    if hasattr(pred_results[0], 'boxes') and len(pred_results[0].boxes) > 0:
                        boxes = pred_results[0].boxes
                        for box in boxes:
                            # 获取坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 获取置信度和类别
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"class{cls_id}"
                            
                            # 绘制边界框和标签
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imwrite(preview_path, img)
                
                # 检查文件存在并更新UI
                if os.path.exists(preview_path):
                    self.log_update.emit(f"预览图像已保存: {preview_path}")
                    self.image_update.emit(preview_path)
                else:
                    self.log_update.emit("预览图像保存失败")
            
            except Exception as e:
                self.log_update.emit(f"创建预览图像时出错: {str(e)}")
                traceback.print_exc()
                
                # 最后的备选方案是显示原始图像
                self.log_update.emit("无法创建预览，显示原始图像")
                self.image_update.emit(sample_img_path)
            
        except Exception as e:
            self.log_update.emit(f"标准验证失败: {str(e)}")
            import traceback
            self.log_update.emit(traceback.format_exc())
            self.log_update.emit("回退到推理模式...")
            self.run_inference_on_images()

    def _apply_custom_names(self, custom_names):
        """应用自定义类名而不直接修改model.names属性"""
        # 存储自定义类名
        self._custom_names = custom_names
        # 备份原始类名
        self._original_names = self.model.names.copy() if hasattr(self.model.names, 'copy') else dict(self.model.names)
        self.log_update.emit(f"已保存原始模型类名，并使用自定义类名进行可视化")
        
        # 定义ModelWrapper类
        class ModelWrapper:
            def __init__(self, model, custom_names):
                self.model = model
                self.custom_names = custom_names
            
            def __getattr__(self, name):
                if name == 'names':
                    return self.custom_names
                return getattr(self.model, name)
                
            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)
        
        # 使用包装器包装模型
        self.wrapped_model = ModelWrapper(self.model, self._custom_names)
        # 保留对原始模型的引用
        self._original_model = self.model
        # 替换模型引用为包装器
        self.model = self.wrapped_model 