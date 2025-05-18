import os
import sys
import time
import traceback
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import cv2
import torch

class InferenceWorker(QObject):
    """Worker for running YOLO model inference in a separate thread."""
    
    # Signals
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    stats_update = pyqtSignal(str)
    image_update = pyqtSignal(str)
    inference_complete = pyqtSignal()
    inference_error = pyqtSignal(str)
    
    def __init__(self, model_path, input_path, output_dir, is_folder_mode=False,
                 conf_thresh=0.25, iou_thresh=0.45, img_size=640, save_results=True):
        """Initialize the inference worker with parameters."""
        super().__init__()
        self.model_path = model_path
        self.input_path = input_path
        self.output_dir = output_dir
        self.is_folder_mode = is_folder_mode
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.save_results = save_results
        self.should_stop = False
        self.model = None
    
    def stop(self):
        """Signal the worker to stop processing."""
        self.should_stop = True
        self.log_update.emit("接收到停止信号，正在停止...")
    
    def run(self):
        """Execute the inference process."""
        try:
            self.log_update.emit(f"开始使用模型进行推理: {os.path.basename(self.model_path)}")
            self.log_update.emit(f"模式: {'文件夹模式' if self.is_folder_mode else '图片模式'}")
            self.log_update.emit(f"输入路径: {self.input_path}")
            self.log_update.emit(f"输出目录: {self.output_dir}")
            self.log_update.emit(f"置信度阈值: {self.conf_thresh}")
            self.log_update.emit(f"IoU阈值: {self.iou_thresh}")
            self.log_update.emit(f"图像尺寸: {self.img_size}")
            
            # Make sure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load YOLO model
            self.load_model()
            
            if self.should_stop:
                self.log_update.emit("推理已停止")
                return
            
            # Run inference based on mode
            if self.is_folder_mode:
                self.run_folder_inference()
            else:
                # Check if input path contains multiple images (separated by semicolons)
                if ';' in self.input_path:
                    self.run_multiple_images_inference()
                else:
                    self.run_single_image_inference()
                
            if not self.should_stop:
                self.inference_complete.emit()
                
        except Exception as e:
            error_msg = f"推理过程中出错: {str(e)}\n{traceback.format_exc()}"
            self.inference_error.emit(error_msg)
    
    def load_model(self):
        """Load the YOLO model."""
        self.log_update.emit("正在加载YOLO模型...")
        
        try:
            # Try to use Ultralytics YOLO
            import ultralytics
            from ultralytics import YOLO
            
            # Log the Ultralytics version
            self.log_update.emit(f"使用Ultralytics版本: {ultralytics.__version__}")
            
            # Check if this is a yolo12 model
            is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
            
            # Load the model with appropriate parameters
            if is_yolo12:
                self.log_update.emit("检测到yolo12模型，使用兼容模式加载")
                self.model = YOLO(self.model_path, task='detect')
            else:
                self.model = YOLO(self.model_path)
            
            # Log model information
            self.log_update.emit(f"模型已加载: {self.model_path}")
            self.log_update.emit(f"模型任务: {self.model.task}")
            self.log_update.emit(f"模型加载成功")
            
            # Check model attributes for compatibility
            self.ultralytics_version = "v8+" if hasattr(self.model, 'predict') else "v5"
            self.log_update.emit(f"检测到API兼容性: Ultralytics {self.ultralytics_version}")
            
        except ImportError:
            self.log_update.emit("未找到Ultralytics包，尝试使用torch hub...")
            try:
                # Fallback to torch hub for YOLOv5
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                self.model.conf = self.conf_thresh
                self.model.iou = self.iou_thresh
                self.model.classes = None  # all classes
                self.model.max_det = 300  # maximum number of detections
                self.ultralytics_version = "torch_hub"
                self.log_update.emit("模型已从torch hub加载")
            except Exception as e:
                raise ValueError(f"加载模型失败: {str(e)}")
    
    def run_single_image_inference(self):
        """Run inference on a single image."""
        self.log_update.emit("对单个图像进行推理...")
        
        # Get image path
        img_path = self.input_path
        
        # Set up output directory
        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Update progress
        self.progress_update.emit(10)  # 10% - Starting
        
        # Run inference
        try:
            self.log_update.emit(f"对图像进行推理: {os.path.basename(img_path)}")
            
            # Check if this is a yolo12 model
            is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
            
            # Try to use the appropriate API
            if is_yolo12:
                # yolo12 API requires specific parameters
                self.log_update.emit("使用yolo12 API进行推理")
                results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            else:
                # Try to use the newer Ultralytics v8 API first
                try:
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                except TypeError as e:
                    # Log error and try with compatible parameters
                    self.log_update.emit(f"API参数错误: {str(e)}, 尝试向下兼容参数")
                    # Use older parameter name (size)
                    results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                
            # Update progress
            self.progress_update.emit(50)  # 50% - Inference done
            
            # Count detections
            detection_counts = {}
            total_detections = 0
            
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                
                # Log detection count
                if total_detections > 0:
                    self.log_update.emit(f"检测到 {total_detections} 个目标")
                    for cls_name, count in detection_counts.items():
                        self.log_update.emit(f"  - {cls_name}: {count} 个")
                else:
                    self.log_update.emit("未检测到任何目标")
            except Exception as e:
                self.log_update.emit(f"统计检测结果时出错: {str(e)}")
            
            # Save result image
            result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
            
            if self.save_results:
                try:
                    result_saved = False
                    
                    # First try with Ultralytics v8 API
                    if hasattr(results[0], 'plot'):
                        # YOLOv8 or yolo12 API
                        self.log_update.emit("使用YOLOv8/yolo12 API绘制检测结果")
                        result_img = results[0].plot()
                        cv2.imwrite(result_path, result_img)
                        self.log_update.emit(f"结果已保存到: {result_path}")
                        result_saved = True
                    # Then try with Ultralytics v5 API
                    elif hasattr(results, 'render'):
                        # YOLOv5 API
                        self.log_update.emit("使用YOLOv5 API绘制检测结果")
                        results.render()  # Updates results.imgs with boxes
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            cv2.imwrite(result_path, results.imgs[0])
                            self.log_update.emit(f"结果已保存到: {result_path}")
                            result_saved = True
                    
                    # Manual save as fallback
                    if not result_saved:
                        self.log_update.emit("使用OpenCV手动绘制检测结果")
                        # Load original image
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_update.emit(f"无法读取图像: {img_path}")
                            return
                            
                        # Draw results on the image (for YOLOv5 format)
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"Class {int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw results on the image (for YOLOv8 format)
                        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                # Get coordinates
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        cv2.imwrite(result_path, img)
                        self.log_update.emit(f"结果已保存到: {result_path}")
                except Exception as e:
                    self.log_update.emit(f"保存结果图像时出错: {str(e)}\n{traceback.format_exc()}")
            
            # Update display with result image
            if os.path.exists(result_path):
                self.image_update.emit(result_path)
            else:
                self.image_update.emit(img_path)  # Fallback to input image
            
            # Update status with detection summary
            stats_text = f"检测到 {total_detections} 个目标\n"
            if detection_counts:
                for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                    stats_text += f"{cls_name}: {count} 个\n"
            self.stats_update.emit(stats_text)
            
            # Update progress to 100%
            self.progress_update.emit(100)
            
        except Exception as e:
            self.log_update.emit(f"推理过程中出错: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def run_folder_inference(self):
        """Run inference on all images in a folder."""
        self.log_update.emit("对文件夹中的图像进行推理...")
        
        # Find all images in the folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            raise ValueError("文件夹中未找到任何图片文件")
        
        self.log_update.emit(f"找到 {len(image_files)} 张图片")
        
        # Create output directory
        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize detection counts
        detection_counts = {}
        total_detections = 0
        
        # Process images
        for i, img_path in enumerate(image_files):
            if self.should_stop:
                self.log_update.emit("推理已停止")
                return
            
            # Update progress
            progress = int((i + 1) / len(image_files) * 100)
            self.progress_update.emit(progress)
            
            # Log current file
            self.log_update.emit(f"处理 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            # Run inference
            try:
                # Check if this is a yolo12 model
                is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
                
                # Try to use the appropriate API
                if is_yolo12:
                    # yolo12 API requires specific parameters
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                else:
                    # First try with newer Ultralytics v8 API
                    try:
                        results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                    except TypeError as e:
                        # Log error and try with compatible parameters
                        self.log_update.emit(f"API参数错误: {str(e)}, 尝试向下兼容参数")
                        # Use older parameter name (size)
                        results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            except Exception as e:
                self.log_update.emit(f"对图像 {os.path.basename(img_path)} 推理失败: {str(e)}")
                continue
            
            # Count detections
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
            except Exception as e:
                self.log_update.emit(f"统计检测结果时出错: {str(e)}")
            
            # Save results
            if self.save_results:
                try:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    result_saved = False
                    
                    # First try with Ultralytics v8 API
                    if hasattr(results[0], 'plot'):
                        # YOLOv8 API
                        result_img = results[0].plot()
                        cv2.imwrite(result_path, result_img)
                        result_saved = True
                    # Then try with Ultralytics v5 API
                    elif hasattr(results, 'render'):
                        # YOLOv5 API
                        results.render()  # Updates results.imgs with boxes
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            cv2.imwrite(result_path, results.imgs[0])
                            result_saved = True
                    
                    # Manual save as fallback
                    if not result_saved:
                        # Load original image
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_update.emit(f"无法读取图像: {img_path}")
                            continue
                            
                        # Draw results on the image (for YOLOv5 format)
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"Class {int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw results on the image (for YOLOv8 format)
                        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                # Get coordinates
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        cv2.imwrite(result_path, img)
                except Exception as e:
                    self.log_update.emit(f"保存结果图像时出错: {str(e)}")
            
            # Update display with current result image (every 10% or last image)
            if i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1:
                if self.save_results:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    if os.path.exists(result_path):
                        self.image_update.emit(result_path)
                    else:
                        self.image_update.emit(img_path)  # Fallback to input image
                else:
                    self.image_update.emit(img_path)
        
        # Update statistics with summary
        stats_text = f"图片总数: {len(image_files)}\n检测到目标总数: {total_detections}\n\n"
        if detection_counts:
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                stats_text += f"{cls_name}: {count}个 ({percentage:.1f}%)\n"
        
        self.stats_update.emit(stats_text)
        
        # Log summary
        self.log_update.emit(f"推理完成! 处理了 {len(image_files)} 张图片，检测到 {total_detections} 个目标")
        self.log_update.emit(f"结果已保存到: {results_dir}")
        
        if detection_counts:
            self.log_update.emit("类别分布:")
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                self.log_update.emit(f"  {cls_name}: {count}个 ({percentage:.1f}%)")
    
    def run_multiple_images_inference(self):
        """Run inference on multiple selected images."""
        self.log_update.emit("对多张图像进行推理...")
        
        # Split input paths
        image_paths = self.input_path.split(';')
        self.log_update.emit(f"共 {len(image_paths)} 张图片需要处理")
        
        # Set up output directory
        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize detection counts
        detection_counts = {}
        total_detections = 0
        
        # Process images
        for i, img_path in enumerate(image_paths):
            if self.should_stop:
                self.log_update.emit("推理已停止")
                return
            
            # Update progress
            progress = int((i + 1) / len(image_paths) * 100)
            self.progress_update.emit(progress)
            
            # Log current file
            self.log_update.emit(f"处理 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Run inference
            try:
                # Check if this is a yolo12 model
                is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
                
                # Try to use the appropriate API
                if is_yolo12:
                    # yolo12 API requires specific parameters
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                else:
                    # First try with newer Ultralytics v8 API
                    try:
                        results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                    except TypeError as e:
                        # Log error and try with compatible parameters
                        self.log_update.emit(f"API参数错误: {str(e)}, 尝试向下兼容参数")
                        # Use older parameter name (size)
                        results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            except Exception as e:
                self.log_update.emit(f"对图像 {os.path.basename(img_path)} 推理失败: {str(e)}")
                continue
            
            # Count detections
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
            except Exception as e:
                self.log_update.emit(f"统计检测结果时出错: {str(e)}")
            
            # Save results
            if self.save_results:
                try:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    result_saved = False
                    
                    # First try with Ultralytics v8 API
                    if hasattr(results[0], 'plot'):
                        # YOLOv8 API
                        result_img = results[0].plot()
                        cv2.imwrite(result_path, result_img)
                        result_saved = True
                    # Then try with Ultralytics v5 API
                    elif hasattr(results, 'render'):
                        # YOLOv5 API
                        results.render()  # Updates results.imgs with boxes
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            cv2.imwrite(result_path, results.imgs[0])
                            result_saved = True
                    
                    # Manual save as fallback
                    if not result_saved:
                        # Load original image
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_update.emit(f"无法读取图像: {img_path}")
                            continue
                            
                        # Draw results on the image (for YOLOv5 format)
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"Class {int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw results on the image (for YOLOv8 format)
                        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                # Get coordinates
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        cv2.imwrite(result_path, img)
                except Exception as e:
                    self.log_update.emit(f"保存结果图像时出错: {str(e)}\n{traceback.format_exc()}")
            
            # Update display with current result image (every 10% or last image)
            if i % max(1, len(image_paths) // 10) == 0 or i == len(image_paths) - 1:
                if self.save_results:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    if os.path.exists(result_path):
                        self.image_update.emit(result_path)
                    else:
                        self.image_update.emit(img_path)  # Fallback to input image
                else:
                    self.image_update.emit(img_path)
        
        # Update statistics with summary
        stats_text = f"图片总数: {len(image_paths)}\n检测到目标总数: {total_detections}\n\n"
        if detection_counts:
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                stats_text += f"{cls_name}: {count}个 ({percentage:.1f}%)\n"
        
        self.stats_update.emit(stats_text)
        
        # Log summary
        self.log_update.emit(f"推理完成! 处理了 {len(image_paths)} 张图片，检测到 {total_detections} 个目标")
        self.log_update.emit(f"结果已保存到: {results_dir}")
        
        if detection_counts:
            self.log_update.emit("类别分布:")
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                self.log_update.emit(f"  {cls_name}: {count}个 ({percentage:.1f}%)") 