import os
import sys
import time
import threading
import torch
from PyQt5.QtCore import QObject, pyqtSignal

class TrainingWorker(QObject):
    """Worker class to handle YOLO model training in a separate thread."""
    
    # Signal definitions
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    training_complete = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, model_type, train_dir, val_dir, output_dir, project_name,
                 dataset_format, batch_size, epochs, img_size, learning_rate, pretrained, model_weights=None, fine_tuning=False):
        """
        Initialize the training worker with parameters.
        
        Args:
            model_type (str): YOLO model type (e.g., 'yolov8n')
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            output_dir (str): Path to save output results
            project_name (str): Project name for output organization
            dataset_format (str): Dataset format ('COCO' or 'VOC')
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            img_size (int): Image size for training
            learning_rate (float): Learning rate
            pretrained (bool): Whether to use pretrained weights
            model_weights (str, optional): Path to custom model weights for initialization
            fine_tuning (bool): Whether to freeze backbone layers and only train detection head
        """
        super().__init__()
        self.model_type = model_type
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_dir = output_dir
        self.project_name = project_name
        self.dataset_format = dataset_format
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.pretrained = pretrained
        self.model_weights = model_weights
        self.fine_tuning = fine_tuning
        
        self._stop_event = threading.Event()
        self._trainer_ref = None  # Reference to the trainer object for direct access
        self._process_ref = None  # Reference to any training process that might be running
    
    def _check_internet_connection(self):
        """
        Check for internet connectivity by attempting to connect to known servers.
        
        Returns:
            bool: True if internet is available, False otherwise
        """
        try:
            # Try to connect to Google's DNS server (should work in most countries)
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            try:
                # Try to connect to Baidu (for users in China)
                socket.create_connection(("220.181.38.148", 80), timeout=3)
                return True
            except OSError:
                pass
        
        # Alternative method: try to resolve a known domain
        try:
            socket.gethostbyname("google.com")
            return True
        except:
            try:
                socket.gethostbyname("baidu.com")
                return True
            except:
                pass
        
        return False
    
    def run(self):
        """Run the training process."""
        try:
            self.log_update.emit(f"Starting training with {self.model_type}")
            print(f"Starting training with {self.model_type}")
            self.log_update.emit(f"Dataset format: {self.dataset_format}")
            print(f"Dataset format: {self.dataset_format}")
            self.log_update.emit(f"Batch size: {self.batch_size}, Image size: {self.img_size}")
            print(f"Batch size: {self.batch_size}, Image size: {self.img_size}")
            self.log_update.emit(f"Learning rate: {self.learning_rate}, Epochs: {self.epochs}")
            print(f"Learning rate: {self.learning_rate}, Epochs: {self.epochs}")
            
            # Check internet connectivity for model downloading
            has_internet = self._check_internet_connection()
            if not has_internet and self.pretrained and not self.model_weights:
                self.log_update.emit("警告：检测到没有互联网连接。若本地没有预训练模型文件，将自动切换到从头训练模式")
            
            # 预加载YOLO模型，这可以避免在训练时重复下载权重
            if not self.model_weights and self.pretrained:
                model_cache_dir = os.path.join(self.output_dir, "model_cache")
                os.makedirs(model_cache_dir, exist_ok=True)
                model_file = os.path.join(model_cache_dir, f"{self.model_type}.pt")
                
                # Check for model in multiple locations with priority
                model_found = False
                
                # Check locations to look for model files
                possible_locations = [
                    # 1. Current directory (highest priority)
                    f"{self.model_type}.pt",
                    # 2. Cache directory
                    model_file,
                    # 3. Common model directories
                    os.path.join("models", f"{self.model_type}.pt"),
                    os.path.join("weights", f"{self.model_type}.pt"),
                    os.path.join("pretrained", f"{self.model_type}.pt"),
                    # 4. Application directory
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{self.model_type}.pt")
                ]
                
                # Look for model file in possible locations
                for location in possible_locations:
                    if os.path.exists(location):
                        self.log_update.emit(f"找到本地模型权重: {location}")
                        # Copy to cache if not already there
                        if location != model_file:
                            try:
                                import shutil
                                shutil.copy(location, model_file)
                                self.log_update.emit(f"已复制模型权重到缓存目录: {model_file}")
                            except Exception as e:
                                self.log_update.emit(f"复制模型到缓存失败，将直接使用原始文件: {str(e)}")
                        self.model_weights = location
                        model_found = True
                        break
                
                # If model not found locally, prepare for download
                if not model_found:
                    self.log_update.emit(f"本地未找到模型文件 {self.model_type}.pt，将尝试自动下载")
                    self.model_weights = None
                
                    # Try to create a placeholder for the download target
                    # This will allow the YOLO loader to download directly to our cache
                    try:
                        # Create an empty file to mark the download location
                        with open(model_file, 'w') as f:
                            f.write("# Placeholder for model download\n")
                        self.log_update.emit(f"已准备下载位置: {model_file}")
                        # We don't set model_weights yet - let YOLO handle the download
                    except Exception as e:
                        self.log_update.emit(f"准备下载位置失败: {str(e)}")
            
            # Create data.yaml file based on dataset format
            yaml_path = self._create_dataset_yaml()
            
            # Check GPU availability
            device = self._check_gpu()
            self.log_update.emit(f"Using device: {device}")
            
            # Import YOLO after checking environment
            # This is done inside the run method to avoid importing in the main thread
            try:
                from ultralytics import YOLO
                # 尝试获取ultralytics版本
                import ultralytics
                ultralytics_version = getattr(ultralytics, '__version__', 'unknown')
                self.log_update.emit(f"Ultralytics YOLO imported successfully (version: {ultralytics_version})")
                
                # 检测是否可以导入Callback类
                has_callback_class = False
                try:
                    from ultralytics.utils.callbacks.base import Callback
                    has_callback_class = True
                    self.log_update.emit("Callback class available")
                except ImportError:
                    self.log_update.emit("Callback class not available, will use function-based callbacks")
                    has_callback_class = False
                
            except ImportError as e:
                self.training_error.emit(f"Failed to import ultralytics: {str(e)}")
                return
            
            # Initialize the model
            try:
                if self.model_weights:
                    # Use specified model weights
                    self.log_update.emit(f"正在加载指定模型权重: {self.model_weights}")
                    model = YOLO(self.model_weights)
                    self.log_update.emit(f"成功加载模型权重: {self.model_weights}")
                elif self.pretrained:
                    # Use pretrained weights
                    self.log_update.emit(f"正在加载预训练权重: {self.model_type}")
                    
                    # Check if it's a YOLO12 model
                    if 'yolo12' in self.model_type.lower():
                        self.log_update.emit(f"检测到YOLO12模型类型: {self.model_type}")
                        try:
                            # For YOLO12, need to specify the correct task
                            model = YOLO(f"{self.model_type}.pt", task='detect')
                            self.log_update.emit(f"成功加载预训练YOLO12模型: {self.model_type}")
                        except Exception as e:
                            error = str(e)
                            self.log_update.emit(f"加载YOLO12预训练模型失败: {error}")
                            
                            # Check if it's a network issue
                            if "not online" in error.lower() or "download failure" in error.lower():
                                self.log_update.emit("检测到网络连接问题，尝试从头开始训练模型")
                                # Fall back to training from scratch
                                model = YOLO(f"{self.model_type}.yaml", task='detect')
                                self.log_update.emit(f"已从头初始化YOLO12模型: {self.model_type}")
                            else:
                                # Re-raise the exception if it's not a network issue
                                raise
                    else:
                        # Handle YOLOv5/YOLOv8 models
                        try:
                            # Standard model loading
                            model = YOLO(f"{self.model_type}.pt")
                            self.log_update.emit(f"成功加载预训练模型: {self.model_type}")
                        except Exception as e:
                            error = str(e)
                            self.log_update.emit(f"加载预训练模型失败: {error}")
                            
                            # Check if it's a network issue
                            if "not online" in error.lower() or "download failure" in error.lower():
                                self.log_update.emit("检测到网络连接问题，尝试从头开始训练模型")
                                # Fall back to training from scratch
                                model = YOLO(f"{self.model_type}.yaml")
                                self.log_update.emit(f"已从头初始化模型: {self.model_type}")
                            else:
                                # Re-raise the exception if it's not a network issue
                                raise
                else:
                    # For training from scratch, use the yaml file of the model architecture
                    self.log_update.emit(f"将从头开始训练模型: {self.model_type}")
                    
                    # Check for model YAML file in multiple locations
                    yaml_found = False
                    yaml_file = None
                    
                    # Possible locations for YAML architecture files
                    yaml_locations = [
                        f"{self.model_type}.yaml",  # Current directory
                        os.path.join("models", f"{self.model_type}.yaml"),
                        os.path.join("configs", f"{self.model_type}.yaml"),
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    f"{self.model_type}.yaml")
                    ]
                    
                    for location in yaml_locations:
                        if os.path.exists(location):
                            yaml_file = location
                            yaml_found = True
                            self.log_update.emit(f"找到本地模型配置文件: {yaml_file}")
                            break
                    
                    try:
                        if 'yolo12' in self.model_type.lower():
                            # YOLO12 requires a different YAML path format
                            # Use found YAML file or rely on Ultralytics default paths
                            if yaml_found:
                                model = YOLO(yaml_file, task='detect')
                            else:
                                model = YOLO(f"{self.model_type}.yaml", task='detect')
                            self.log_update.emit(f"已从头初始化YOLO12模型: {self.model_type}")
                        else:
                            # Standard YOLO model
                            if yaml_found:
                                model = YOLO(yaml_file)
                            else:
                                model = YOLO(f"{self.model_type}.yaml")
                            self.log_update.emit(f"已从头初始化模型: {self.model_type}")
                    except Exception as e:
                        error = str(e)
                        self.log_update.emit(f"加载模型配置文件失败: {error}")
                        
                        # Try to handle common architecture file errors
                        if "Cannot find" in error or "No such file" in error:
                            # Try to fall back to similar model sizes
                            fallback_model = None
                            
                            # Map model types to fallbacks
                            if self.model_type.endswith("n"):
                                fallback_model = "yolov8n"
                            elif self.model_type.endswith("s"):
                                fallback_model = "yolov8s"
                            elif self.model_type.endswith("m"):
                                fallback_model = "yolov8m"
                            elif self.model_type.endswith("l"):
                                fallback_model = "yolov8l"
                            elif self.model_type.endswith("x"):
                                fallback_model = "yolov8x"
                            
                            if fallback_model:
                                self.log_update.emit(f"尝试使用替代模型架构: {fallback_model}")
                                model = YOLO(f"{fallback_model}.yaml")
                                self.log_update.emit(f"已使用替代模型架构: {fallback_model}")
                            else:
                                # Last resort fallback
                                self.log_update.emit("使用标准YOLOv8n架构作为后备方案")
                                model = YOLO("yolov8n.yaml")
                        else:
                            # Re-raise if not a missing file issue
                            raise
                
                # Log model information
                task_name = getattr(model, 'task', 'detect')
                self.log_update.emit(f"模型任务类型: {task_name}")
                
                # Apply fine-tuning mode if requested
                if self.fine_tuning:
                    self.log_update.emit("启用微调模式: 冻结检测头之前的所有参数，仅更新检测头参数")
                    
                    # Access model's pytorch module
                    pytorch_model = model.model
                    
                    # First, we'll freeze all parameters
                    for param in pytorch_model.parameters():
                        param.requires_grad = False
                        
                    # Then, unfreeze only the detection head layers
                    # For YOLOv8, the detection head is in the 'model.model.model[-1]' (detection module)
                    detection_head = None
                    
                    # YOLOv8 models may have different structures, so try different paths
                    try:
                        # yolo12 specific structure handling
                        if 'yolo12' in self.model_type:
                            self.log_update.emit("检测到yolo12结构，正在识别检测头...")
                            
                            # yolo12 has a different structure than YOLOv8
                            # Try to identify the detection head by name or position
                            if hasattr(pytorch_model, 'model'):
                                if hasattr(pytorch_model.model, 'detect'):
                                    # Direct detect module
                                    detection_head = pytorch_model.model.detect
                                    self.log_update.emit("找到yolo12检测头: model.detect")
                                elif hasattr(pytorch_model.model, 'head'):
                                    # Head module
                                    detection_head = pytorch_model.model.head
                                    self.log_update.emit("找到yolo12检测头: model.head")
                                else:
                                    # Try to locate by position (last modules)
                                    layers = list(pytorch_model.model.children())
                                    # Assume the last 1-2 modules are detection related
                                    detection_head = layers[-1]
                                    self.log_update.emit(f"使用yolo12最后一层作为检测头: {detection_head.__class__.__name__}")
                            else:
                                self.log_update.emit("无法识别yolo12结构，将尝试常规方法")
                        
                        # Standard YOLOv8 structure: typically the last module is the detection head
                        elif hasattr(pytorch_model, 'model') and hasattr(pytorch_model.model, 'model'):
                            detection_head = pytorch_model.model.model[-1]
                            self.log_update.emit("检测到YOLOv8结构，已找到检测头")
                        else:
                            # For other structures, try to identify the detection head by name
                            for name, module in pytorch_model.named_children():
                                if 'detect' in name.lower() or 'head' in name.lower():
                                    detection_head = module
                                    self.log_update.emit(f"根据名称找到检测头: {name}")
                                    break
                            
                            # If we still can't find it, try last layer as fallback
                            if detection_head is None:
                                # Get the last layer as fallback
                                layers = list(pytorch_model.children())
                                detection_head = layers[-1]
                                self.log_update.emit("使用模型最后一层作为检测头进行微调")
                        
                        # Unfreeze the detection head parameters
                        if detection_head:
                            for param in detection_head.parameters():
                                param.requires_grad = True
                            
                            # Count trainable parameters
                            trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
                            total_params = sum(p.numel() for p in pytorch_model.parameters())
                            self.log_update.emit(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
                            self.log_update.emit(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
                        else:
                            self.log_update.emit("警告: 无法找到检测头，微调模式可能无效")
                    except Exception as e:
                        error_msg = f"设置微调模式时出错: {str(e)}"
                        self.log_update.emit(error_msg)
                        self.log_update.emit("将继续训练但微调设置可能未成功应用")
                
            except Exception as e:
                self.training_error.emit(f"Failed to initialize model: {str(e)}")
                return
            
            # Start training
            try:
                # 创建保存指标的目录
                metrics_dir = os.path.join(self.output_dir, self.project_name)
                
                # 设置进度更新的监控线程
                stop_flag = threading.Event()
                
                def progress_monitor():
                    last_metrics_time = 0
                    metrics_file = None
                    
                    # 寻找可能的指标文件路径
                    def find_metrics_file():
                        # 检查最近创建的run目录
                        run_dirs = []
                        if os.path.exists(metrics_dir):
                            for d in os.listdir(metrics_dir):
                                full_path = os.path.join(metrics_dir, d)
                                if os.path.isdir(full_path) and d.startswith("exp") or d.startswith("train"):
                                    run_dirs.append((os.path.getmtime(full_path), full_path))
                        
                        # 按修改时间排序，获取最新的目录
                        if run_dirs:
                            latest_dir = sorted(run_dirs, reverse=True)[0][1]
                            # 检查CSV文件
                            csv_path = os.path.join(latest_dir, "results.csv")
                            if os.path.exists(csv_path):
                                return csv_path
                        return None
                    
                    # 初始进度更新 - 不再使用固定延迟
                    self.progress_update.emit(5)
                    self.log_update.emit("加载和准备环境...")
                    
                    # 主循环监控训练进度
                    while not stop_flag.is_set() and not self._stop_event.is_set():
                        # 尝试找到指标文件
                        if metrics_file is None:
                            metrics_file = find_metrics_file()
                        
                        # 如果找到了指标文件，读取并显示最新指标
                        if metrics_file and os.path.exists(metrics_file):
                            current_time = os.path.getmtime(metrics_file)
                            
                            # 只有当文件更新时才读取
                            if current_time > last_metrics_time:
                                last_metrics_time = current_time
                                try:
                                    with open(metrics_file, 'r') as f:
                                        lines = f.readlines()
                                        if len(lines) > 1:  # 至少有标题行和一行数据
                                            last_line = lines[-1].strip()
                                            header = lines[0].strip().split(',')
                                            values = last_line.split(',')
                                            
                                            # 解析指标
                                            metrics = {}
                                            for i, key in enumerate(header):
                                                if i < len(values):
                                                    try:
                                                        metrics[key] = float(values[i])
                                                    except ValueError:
                                                        metrics[key] = values[i]
                                            
                                            # 更新进度
                                            if 'epoch' in metrics and 'epochs' in metrics:
                                                epoch = metrics['epoch']
                                                epochs = metrics['epochs']
                                                progress = int((epoch / epochs) * 100)
                                                self.progress_update.emit(progress)
                                            
                                            # 组织指标信息
                                            info_text = f"Epoch: {metrics.get('epoch', '?')}/{metrics.get('epochs', '?')}\n"
                                            
                                            # 添加损失指标
                                            losses = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"]
                                            info_text += "损失指标:\n"
                                            for loss in losses:
                                                if loss in metrics:
                                                    info_text += f"  {loss}: {metrics[loss]:.4f}\n"
                                            
                                            # 添加精度指标
                                            accuracies = ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"]
                                            info_text += "精度指标:\n"
                                            for acc in accuracies:
                                                if acc in metrics:
                                                    info_text += f"  {acc}: {metrics[acc]:.4f}\n"
                                            
                                            # 输出完整信息
                                            self.log_update.emit(info_text)
                                            print(info_text)  # Direct stdout output
                                except Exception as e:
                                    error_msg = f"读取指标文件出错: {str(e)}"
                                    self.log_update.emit(error_msg)
                                    print(error_msg, file=sys.stderr)  # Direct stderr output
                        
                        # 休眠时间缩短，更快地检查更新
                        time.sleep(0.5)
                
                # 记录开始时间
                start_time = time.time()
                
                # 启动监控线程
                self.log_update.emit("启动进度监控...")
                monitor_thread = threading.Thread(target=progress_monitor)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # 尝试设置自定义stdout捕获类，以便更实时地获取训练输出
                class StdoutCapture:
                    def __init__(self, worker):
                        self.worker = worker
                        self.original_stdout = sys.stdout
                        self.original_stderr = sys.stderr
                        self.buffer = ""

                    def write(self, text):
                        # 写入原始流
                        self.original_stdout.write(text)
                        self.original_stdout.flush()
                        
                        # 添加到缓冲区
                        self.buffer += text
                        
                        # 如果有完整行，则发送到UI
                        if '\n' in text:
                            lines = self.buffer.split('\n')
                            for line in lines[:-1]:  # 处理除最后一个可能不完整的行外的所有行
                                if line.strip():  # 如果行不为空
                                    # 只处理训练相关信息
                                    if "Epoch" in line and ("GPU_mem" in line or "box_loss" in line):
                                        self.worker.log_update.emit(line)
                            
                            # 保留最后一个不完整的行
                            self.buffer = lines[-1] if lines else ""
                    
                    def flush(self):
                        self.original_stdout.flush()
                        
                    def __enter__(self):
                        sys.stdout = self
                        return self
                    
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        sys.stdout = self.original_stdout
                        
                # 创建捕获实例
                stdout_capture = StdoutCapture(self)
                
                # 创建通用回调函数，支持任何版本的ultralytics
                def on_train_batch_end_fn(trainer=None):
                    # 在每个训练批次结束时检查停止标志
                    if self._stop_event.is_set():
                        self.log_update.emit("检测到停止信号，正在中断训练...")
                        if trainer:
                            self._trainer_ref = trainer  # Store reference to trainer
                            # 尝试停止训练循环
                            if hasattr(trainer, 'epoch_progress'):
                                try:
                                    trainer.epoch_progress.close()  # 关闭进度条
                                except:
                                    pass
                            if hasattr(trainer, 'stop'):
                                trainer.stop = True
                        return False  # 返回False以停止训练循环
                    return True
                
                def on_train_epoch_end_fn(trainer=None):
                    # 在每个epoch结束时检查停止标志
                    if self._stop_event.is_set():
                        self.log_update.emit("检测到停止信号，正在中断训练...")
                        if trainer:
                            self._trainer_ref = trainer  # Store reference to trainer
                            if hasattr(trainer, 'stop'):
                                trainer.stop = True
                        return False  # 返回False以停止训练循环
                    return True
                
                # 添加新的回调函数，用于设置进程参考
                def on_train_start_fn(trainer=None):
                    if trainer:
                        self._trainer_ref = trainer  # Store reference to trainer
                        self.log_update.emit("训练开始，已捕获训练器引用")
                    import threading
                    self._process_ref = threading.current_thread()
                    return True
                
                # 创建带有回调的自定义训练参数
                train_args = {
                    'data': yaml_path,
                    'epochs': self.epochs,
                    'batch': self.batch_size,
                    'imgsz': self.img_size,
                    'project': self.project_name,
                    'name': time.strftime("%Y%m%d-%H%M%S"),
                    'lr0': self.learning_rate,
                    'device': device,
                    'exist_ok': True,
                    'save_dir': self.output_dir,
                    'plots': True
                }
                
                # 根据不同的ultralytics版本，尝试不同的回调方式
                self.log_update.emit("配置训练参数和回调...")
                
                results = None
                
                # 使用不同的回调方法尝试训练
                if has_callback_class:
                    # 方法1: 使用基于类的回调方式
                    try:
                        self.log_update.emit("使用基于类的回调方式")
                        
                        # 创建一个自定义的回调类，通过闭包引用worker对象
                        class CustomStopCallback(Callback):
                            def __init__(self_callback, stop_event, worker):
                                self_callback.stop_event = stop_event
                                self_callback.worker = worker
                            
                            def on_train_start(self_callback, trainer):
                                self_callback.worker._trainer_ref = trainer
                                self_callback.worker.log_update.emit("训练开始，已捕获训练器引用")
                                import threading
                                self_callback.worker._process_ref = threading.current_thread()
                                return True
                            
                            def on_train_batch_end(self_callback, trainer):
                                return on_train_batch_end_fn(trainer)
                            
                            def on_train_epoch_end(self_callback, trainer):
                                return on_train_epoch_end_fn(trainer)
                        
                        # 创建回调实例
                        callback = CustomStopCallback(self._stop_event, self)
                        train_args['callbacks'] = [callback]
                        
                        # 使用捕获器运行训练
                        with stdout_capture:
                            # 使用类回调进行训练
                            self.log_update.emit("开始训练，第一个epoch可能较慢，因为需要进行初始化和缓存")
                            results = model.train(**train_args)
                    except Exception as e:
                        self.log_update.emit(f"使用类回调失败: {str(e)}")
                        # 类方法失败时，继续尝试函数方法
                
                # 如果基于类的回调失败或不可用，尝试函数回调
                if results is None:
                    # 方法2: 使用基于函数的回调方式
                    try:
                        self.log_update.emit("使用函数回调方式 (on_train_batch_end)")
                        # 使用捕获器运行训练 - 函数回调方式
                        with stdout_capture:
                            # 使用函数回调进行训练
                            train_args['callbacks'] = {
                                'on_train_start': on_train_start_fn,
                                'on_train_batch_end': on_train_batch_end_fn,
                                'on_train_epoch_end': on_train_epoch_end_fn
                            }
                            self.log_update.emit("开始训练，第一个epoch可能较慢，因为需要进行初始化和缓存")
                            results = model.train(**train_args)
                    except Exception as e:
                        self.log_update.emit(f"函数回调方式1失败: {str(e)}")
                        
                        # 方法3: 使用另一种回调格式
                        try:
                            self.log_update.emit("使用函数回调方式 (on_fit_epoch_end)")
                            train_args['callbacks'] = {
                                'on_fit_start': lambda trainer: on_train_start_fn(trainer),
                                'on_fit_epoch_end': lambda trainer: not self._stop_event.is_set(),
                                'on_fit_batch_end': lambda trainer: not self._stop_event.is_set()
                            }
                            results = model.train(**train_args)
                        except Exception as e:
                            self.log_update.emit(f"函数回调方式2失败: {str(e)}")
                            
                            # 方法4: 无回调训练
                            self.log_update.emit("尝试无回调训练")
                            if 'callbacks' in train_args:
                                del train_args['callbacks']
                            results = model.train(**train_args)
                
                # 训练完成，停止监控线程
                stop_flag.set()
                
                # 检查结果并更新UI
                if self._stop_event.is_set():
                    self.log_update.emit("训练被用户中止")
                    self.training_complete.emit()
                else:
                    if results is not None and hasattr(results, 'metrics'):
                        metrics = results.metrics
                        self.log_update.emit(f"训练完成! 最终结果:")
                        if hasattr(metrics, 'box_loss'):
                            self.log_update.emit(f"box_loss: {metrics.box_loss:.4f}")
                        if hasattr(metrics, 'cls_loss'):
                            self.log_update.emit(f"cls_loss: {metrics.cls_loss:.4f}")
                        if hasattr(metrics, 'map50'):
                            self.log_update.emit(f"mAP50: {metrics.map50:.4f}")
                    
                    self.log_update.emit("训练成功完成!")
                    self.progress_update.emit(100)
                    self.training_complete.emit()
                
            except Exception as e:
                if self._stop_event.is_set():
                    self.log_update.emit("训练已被用户中止")
                    self.training_complete.emit()
                else:
                    self.training_error.emit(f"训练错误: {str(e)}")
        
        except Exception as e:
            self.training_error.emit(f"意外错误: {str(e)}")
    
    def stop(self):
        """Stop the training process immediately."""
        self._stop_event.set()
        self.log_update.emit("收到停止信号，立即中断训练...")
        
        # Attempt to terminate the training more aggressively
        if self._trainer_ref is not None:
            try:
                # Try all possible ways to forcibly stop the trainer
                if hasattr(self._trainer_ref, 'stop'):
                    self._trainer_ref.stop = True
                if hasattr(self._trainer_ref, 'epoch_progress') and hasattr(self._trainer_ref.epoch_progress, 'close'):
                    self._trainer_ref.epoch_progress.close()
                if hasattr(self._trainer_ref, 'stopper') and hasattr(self._trainer_ref.stopper, 'run'):
                    self._trainer_ref.stopper.possible_stop = True
                self.log_update.emit("已发送终止信号到训练器")
            except Exception as e:
                self.log_update.emit(f"尝试终止训练器时出错: {str(e)}")
        
        # If we have a training process, attempt to terminate it more forcefully
        if self._process_ref is not None:
            try:
                import signal
                import ctypes
                import os
                
                if hasattr(self._process_ref, 'terminate'):
                    self._process_ref.terminate()
                    self.log_update.emit("已强制终止训练进程")
                elif isinstance(self._process_ref, threading.Thread) and self._process_ref.is_alive():
                    # This is a more aggressive approach for Python threads
                    if hasattr(threading, '_async_raise'):
                        threading._async_raise(self._process_ref.ident, SystemExit)
                    self.log_update.emit("已尝试强制终止训练线程")
            except Exception as e:
                self.log_update.emit(f"尝试强制终止训练时出错: {str(e)}")
        
        # Signal that training was stopped by user
        threading.Thread(target=self._emit_training_complete, daemon=True).start()
    
    def _emit_training_complete(self):
        """Emit training complete signal after a short delay to allow cleanup"""
        time.sleep(0.5)  # Short delay to allow other operations to complete
        self.training_complete.emit()
    
    def _check_gpu(self):
        """Check if CUDA is available and return appropriate device."""
        if torch.cuda.is_available():
            return 0  # Use first GPU
        else:
            self.log_update.emit("CUDA not available, using CPU")
            return 'cpu'
    
    def _create_dataset_yaml(self):
        """
        Create the dataset YAML file based on the selected format.
        
        Returns:
            str: Path to the created YAML file
        """
        self.log_update.emit("准备训练数据配置...")
        
        os.makedirs(os.path.join(self.output_dir, "datasets"), exist_ok=True)
        yaml_path = os.path.join(self.output_dir, "datasets", "data.yaml")
        
        # 检查是否存在缓存的YAML文件，而且训练目录没有变化
        cache_info_path = os.path.join(self.output_dir, "datasets", "cache_info.txt")
        if os.path.exists(yaml_path) and os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, 'r') as f:
                    cached_paths = f.read().strip().split('\n')
                
                if len(cached_paths) >= 2 and cached_paths[0] == self.train_dir and cached_paths[1] == self.val_dir:
                    self.log_update.emit("使用缓存的数据集配置...")
                    return yaml_path
            except:
                pass
        
        # 预处理和验证路径
        train_dir = self._normalize_path(self.train_dir)
        val_dir = self._normalize_path(self.val_dir)
        
        # 检查训练和验证目录是否存在
        train_exists = os.path.exists(train_dir)
        val_exists = os.path.exists(val_dir)
        
        if not train_exists:
            self.log_update.emit(f"警告: 训练目录不存在: {train_dir}")
        
        if not val_exists:
            self.log_update.emit(f"警告: 验证目录不存在: {val_dir}")
            # 如果验证目录不存在，使用训练目录替代
            self.log_update.emit("将使用训练目录作为验证数据")
            val_dir = train_dir
            
        # Handle different dataset formats
        if self.dataset_format == "YOLO":
            # For YOLO format, we look for data.yaml in the dataset directory
            possible_yaml_paths = [
                os.path.join(self.train_dir, "data.yaml"),
                os.path.join(os.path.dirname(self.train_dir), "data.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(self.train_dir)), "data.yaml")
            ]
            
            for path in possible_yaml_paths:
                if os.path.exists(path):
                    self.log_update.emit(f"找到YOLO格式数据集配置文件: {path}")
                    # 复制原始YAML到我们的输出目录
                    import shutil
                    shutil.copy(path, yaml_path)
                    
                    # 读取并修改路径（如果需要）
                    self._update_paths_in_yaml(path, yaml_path)
                    
                    # 保存缓存信息
                    try:
                        with open(cache_info_path, 'w') as f:
                            f.write(f"{self.train_dir}\n{self.val_dir}")
                        self.log_update.emit("已缓存数据集信息，下次训练将加速初始化")
                    except Exception as e:
                        self.log_update.emit(f"保存缓存信息失败: {str(e)}")
                    
                    return yaml_path
            
            # 如果没有找到现有的yaml，创建一个
            self.log_update.emit("未找到YOLO格式数据集配置文件，将创建新文件")
            
            # 检查目录结构是否标准YOLO结构
            train_images_dir = train_dir
            train_labels_dir = None
            val_images_dir = val_dir
            val_labels_dir = None
            
            # 检查是否是标准的YOLO目录结构 (根目录/images/train, 根目录/labels/train)
            if os.path.basename(train_dir) == 'train' and os.path.basename(os.path.dirname(train_dir)) == 'images':
                base_dir = os.path.dirname(os.path.dirname(train_dir))
                possible_labels_dir = os.path.join(base_dir, 'labels', 'train')
                if os.path.exists(possible_labels_dir):
                    train_labels_dir = possible_labels_dir
                    self.log_update.emit(f"找到匹配的训练标签目录: {train_labels_dir}")
            
            if os.path.basename(val_dir) == 'val' and os.path.basename(os.path.dirname(val_dir)) == 'images':
                base_dir = os.path.dirname(os.path.dirname(val_dir))
                possible_labels_dir = os.path.join(base_dir, 'labels', 'val')
                if os.path.exists(possible_labels_dir):
                    val_labels_dir = possible_labels_dir
                    self.log_update.emit(f"找到匹配的验证标签目录: {val_labels_dir}")
            
            # 如果没有找到标准结构，尝试搜索标签文件
            if not train_labels_dir:
                self.log_update.emit("尝试查找训练标签目录...")
                # 检查常见的可能标签目录
                possible_label_dirs = [
                    os.path.join(os.path.dirname(train_dir), 'labels'),
                    os.path.join(os.path.dirname(train_dir), 'labels', 'train'),
                    os.path.join(train_dir, '..', 'labels'),
                    os.path.join(train_dir, '..', 'labels', 'train'),
                    os.path.join(os.path.dirname(os.path.dirname(train_dir)), 'labels', 'train')
                ]
                
                for label_dir in possible_label_dirs:
                    if os.path.exists(label_dir) and os.listdir(label_dir):
                        train_labels_dir = label_dir
                        self.log_update.emit(f"找到可能的训练标签目录: {train_labels_dir}")
                        break
            
            if not val_labels_dir:
                self.log_update.emit("尝试查找验证标签目录...")
                # 检查常见的可能标签目录
                possible_label_dirs = [
                    os.path.join(os.path.dirname(val_dir), 'labels'),
                    os.path.join(os.path.dirname(val_dir), 'labels', 'val'),
                    os.path.join(val_dir, '..', 'labels'),
                    os.path.join(val_dir, '..', 'labels', 'val'),
                    os.path.join(os.path.dirname(os.path.dirname(val_dir)), 'labels', 'val')
                ]
                
                for label_dir in possible_label_dirs:
                    if os.path.exists(label_dir) and os.listdir(label_dir):
                        val_labels_dir = label_dir
                        self.log_update.emit(f"找到可能的验证标签目录: {val_labels_dir}")
                        break
            
            # 验证找到的标签目录是否包含与图片匹配的标签
            if train_labels_dir and os.path.exists(train_images_dir):
                # 在训练图像目录中找到图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                train_images = []
                
                for root, _, files in os.walk(train_images_dir):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            train_images.append(os.path.join(root, file))
                
                if train_images:
                    # 检查是否有匹配的标签文件
                    matching_labels = 0
                    for img_file in train_images[:10]:  # 只检查前10个样本
                        img_basename = os.path.splitext(os.path.basename(img_file))[0]
                        label_file = os.path.join(train_labels_dir, f"{img_basename}.txt")
                        if os.path.exists(label_file):
                            matching_labels += 1
                    
                    if matching_labels > 0:
                        self.log_update.emit(f"验证训练标签: 找到 {matching_labels}/10 个匹配的标签文件")
                    else:
                        self.log_update.emit("警告: 训练标签目录中没有找到与图片匹配的标签文件")
                        train_labels_dir = None
            
            # 验证验证集标签目录
            if val_labels_dir and os.path.exists(val_images_dir):
                # 在验证图像目录中找到图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                val_images = []
                
                for root, _, files in os.walk(val_images_dir):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            val_images.append(os.path.join(root, file))
                
                if val_images:
                    # 检查是否有匹配的标签文件
                    matching_labels = 0
                    for img_file in val_images[:10]:  # 只检查前10个样本
                        img_basename = os.path.splitext(os.path.basename(img_file))[0]
                        label_file = os.path.join(val_labels_dir, f"{img_basename}.txt")
                        if os.path.exists(label_file):
                            matching_labels += 1
                    
                    if matching_labels > 0:
                        self.log_update.emit(f"验证验证集标签: 找到 {matching_labels}/10 个匹配的标签文件")
                    else:
                        self.log_update.emit("警告: 验证集标签目录中没有找到与图片匹配的标签文件")
                        val_labels_dir = None
            
            # 如果我们需要创建一个新的数据集结构
            if train_labels_dir is None or val_labels_dir is None:
                self.log_update.emit("未找到完整的标签结构，将尝试建立新的YOLO格式数据集")
                
                # 创建YOLO格式的目录结构
                yolo_dataset_dir = os.path.join(self.output_dir, "yolo_dataset")
                os.makedirs(os.path.join(yolo_dataset_dir, "images", "train"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dataset_dir, "images", "val"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dataset_dir, "labels", "train"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dataset_dir, "labels", "val"), exist_ok=True)
                
                # 后续可以添加代码来复制和准备数据到这个结构
                self.log_update.emit(f"创建了标准YOLO目录结构: {yolo_dataset_dir}")
                
                # 但目前我们暂时使用原始目录
                train_path = os.path.dirname(train_dir)
                train_rel = os.path.basename(train_dir)
                val_path = os.path.dirname(val_dir)
                val_rel = os.path.basename(val_dir)
            else:
                # 使用找到的目录结构
                # 找出训练目录和标签目录的共同父目录
                train_path = self._find_common_parent(train_images_dir, train_labels_dir)
                # 获取相对路径
                train_rel = os.path.relpath(train_images_dir, train_path)
                val_path = self._find_common_parent(val_images_dir, val_labels_dir)
                val_rel = os.path.relpath(val_images_dir, val_path)
                
                self.log_update.emit(f"使用现有的目录结构，基础路径: {train_path}")
                self.log_update.emit(f"训练图像相对路径: {train_rel}")
                self.log_update.emit(f"验证图像相对路径: {val_rel}")
            
            # 获取类名
            class_names = self._get_class_names()
            
            # 创建配置文件
            with open(yaml_path, 'w') as f:
                f.write(f"# Dataset configuration for YOLO format\n")
                f.write(f"path: {train_path}\n")
                f.write(f"train: {train_rel}\n")
                f.write(f"val: {val_rel}\n")
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")
            
            # 验证生成的YAML是否有效
            self._validate_yaml(yaml_path)
            
            self.log_update.emit(f"已创建YOLO格式数据集配置文件: {yaml_path}")
            
            # 保存缓存信息
            try:
                with open(cache_info_path, 'w') as f:
                    f.write(f"{self.train_dir}\n{self.val_dir}")
                self.log_update.emit("已缓存数据集信息，下次训练将加速初始化")
            except Exception as e:
                self.log_update.emit(f"保存缓存信息失败: {str(e)}")
            
            return yaml_path
            
        else:
            # 处理COCO和VOC格式
            # Get class names based on the dataset format
            class_names = self._get_class_names()
            
            with open(yaml_path, 'w') as f:
                f.write(f"# Dataset configuration\n")
                f.write(f"path: {os.path.dirname(self.train_dir)}\n")
                f.write(f"train: {os.path.basename(self.train_dir)}\n")
                f.write(f"val: {os.path.basename(self.val_dir)}\n")
                
                # Write class names
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")
            
            self.log_update.emit(f"Created dataset configuration at {yaml_path}")
            
            # 保存缓存信息
            try:
                with open(cache_info_path, 'w') as f:
                    f.write(f"{self.train_dir}\n{self.val_dir}")
                self.log_update.emit("已缓存数据集信息，下次训练将加速初始化")
            except Exception as e:
                self.log_update.emit(f"保存缓存信息失败: {str(e)}")
            
            return yaml_path
    
    def _update_paths_in_yaml(self, src_yaml, dst_yaml):
        """更新YAML文件中的路径以适应当前环境"""
        import yaml
        
        try:
            # 读取原始YAML
            with open(src_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # 规范化训练和验证目录
            train_dir = self._normalize_path(self.train_dir)
            val_dir = self._normalize_path(self.val_dir)
            
            # 检查验证目录是否存在
            if not os.path.exists(val_dir):
                self.log_update.emit(f"警告: 验证目录不存在: {val_dir}")
                self.log_update.emit("将使用训练目录作为验证数据")
                val_dir = train_dir
            
            # 检查并更新路径
            if 'path' in data:
                # 如果原始路径不存在，改为使用当前路径
                original_path = data['path']
                if not os.path.exists(original_path):
                    data['path'] = os.path.dirname(train_dir)
                    self.log_update.emit(f"更新数据集路径: {original_path} -> {data['path']}")
            else:
                data['path'] = os.path.dirname(train_dir)
            
            # 确保train和val字段存在和正确
            data['train'] = os.path.basename(train_dir)
            data['val'] = os.path.basename(val_dir)
            
            # 确保路径格式正确(不要有多个连续的斜杠)
            data['path'] = self._normalize_path(data['path'])
            
            # 写入更新后的YAML
            with open(dst_yaml, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            # 验证生成的YAML文件
            self._validate_yaml(dst_yaml)
            
        except Exception as e:
            self.log_update.emit(f"更新YAML文件失败: {str(e)}，将创建新文件")
            # 如果失败，创建新文件
            class_names = self._get_class_names()
            with open(dst_yaml, 'w') as f:
                f.write(f"# Dataset configuration\n")
                f.write(f"path: {os.path.dirname(train_dir)}\n")
                f.write(f"train: {os.path.basename(train_dir)}\n")
                f.write(f"val: {os.path.basename(val_dir)}\n")
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")
            
            # 验证生成的YAML文件
            self._validate_yaml(dst_yaml)
    
    def _normalize_path(self, path):
        """规范化路径，确保路径格式正确"""
        # 替换多个连续的斜杠为单个斜杠
        path = path.replace('///', '/').replace('//', '/')
        
        # 确保Windows路径使用正确的斜杠格式
        if os.name == 'nt':
            path = path.replace('/', '\\')
            # 修复可能出现的多个反斜杠问题
            while '\\\\' in path:
                path = path.replace('\\\\', '\\')
        
        return path
    
    def _validate_yaml(self, yaml_path):
        """验证YAML文件中的路径是否有效"""
        import yaml
        try:
            # 读取YAML文件
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 检查基础路径是否存在
            base_path = data.get('path', '')
            if not base_path or not os.path.exists(base_path):
                self.log_update.emit(f"警告: YAML中的基础路径不存在: {base_path}")
                
                # 尝试修复基础路径
                if 'train' in data:
                    train_rel = data['train']
                    possible_base = self._find_valid_base_path(train_rel)
                    if possible_base:
                        data['path'] = possible_base
                        self.log_update.emit(f"自动修复基础路径为: {possible_base}")
                        
                        # 重新写入YAML
                        with open(yaml_path, 'w') as f:
                            yaml.dump(data, f, default_flow_style=False)
            
            # 检查训练和验证路径是否存在
            if 'path' in data and os.path.exists(data['path']):
                if 'train' in data:
                    train_path = os.path.join(data['path'], data['train'])
                    if not os.path.exists(train_path):
                        self.log_update.emit(f"警告: 训练路径不存在: {train_path}")
                
                if 'val' in data:
                    val_path = os.path.join(data['path'], data['val'])
                    if not os.path.exists(val_path):
                        self.log_update.emit(f"警告: 验证路径不存在: {val_path}")
                        
                        # 如果验证路径不存在但训练路径存在，使用训练路径代替
                        if 'train' in data and os.path.exists(os.path.join(data['path'], data['train'])):
                            data['val'] = data['train']
                            self.log_update.emit(f"自动设置验证集与训练集相同: {data['train']}")
                            
                            # 重新写入YAML
                            with open(yaml_path, 'w') as f:
                                yaml.dump(data, f, default_flow_style=False)
        
        except Exception as e:
            self.log_update.emit(f"验证YAML文件失败: {str(e)}")
    
    def _find_valid_base_path(self, train_rel):
        """尝试找到有效的基础路径"""
        # 从当前工作目录开始
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, train_rel)):
            return cwd
        
        # 从训练目录的父级目录尝试
        train_dir = self._normalize_path(self.train_dir)
        parent_dir = os.path.dirname(train_dir)
        if os.path.exists(os.path.join(parent_dir, train_rel)):
            return parent_dir
        
        # 从训练目录本身尝试
        if os.path.basename(train_dir) == train_rel:
            return os.path.dirname(train_dir)
        
        return None
    
    def _get_class_names(self):
        """
        Extract class names from the dataset.
        
        Returns:
            list: List of class names
        """
        if self.dataset_format == "YOLO":
            # 对于YOLO格式，尝试从classes.txt或data.yaml文件中获取类名
            class_names = self._get_yolo_class_names()
        elif self.dataset_format == "COCO":
            # For COCO, we would parse the annotations JSON file
            class_names = self._get_coco_class_names()
        elif self.dataset_format == "VOC":
            # For VOC, we would look for the labels in the annotation XML files
            class_names = self._get_voc_class_names()
        else:
            class_names = ['class0', 'class1']
        
        # 如果没有找到类名，使用默认名称
        if not class_names:
            self.log_update.emit("警告: 无法确定类名，使用默认值")
            class_names = ['class0', 'class1']
        
        return class_names
    
    def _get_yolo_class_names(self):
        """从YOLO格式数据集中提取类名"""
        class_names = []
        
        # 首先尝试从data.yaml文件中获取
        possible_yaml_paths = [
            os.path.join(self.train_dir, "data.yaml"),
            os.path.join(os.path.dirname(self.train_dir), "data.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(self.train_dir)), "data.yaml")
        ]
        
        for yaml_path in possible_yaml_paths:
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if 'names' in data:
                        self.log_update.emit(f"从YAML文件加载类名: {yaml_path}")
                        return data['names']
                except Exception as e:
                    self.log_update.emit(f"从YAML读取类名失败: {str(e)}")
        
        # 然后尝试从classes.txt文件中获取
        possible_class_files = [
            os.path.join(self.train_dir, "classes.txt"),
            os.path.join(os.path.dirname(self.train_dir), "classes.txt"),
            os.path.join(os.path.dirname(os.path.dirname(self.train_dir)), "classes.txt")
        ]
        
        for class_file in possible_class_files:
            if os.path.exists(class_file):
                try:
                    with open(class_file, 'r') as f:
                        class_names = [line.strip() for line in f.readlines() if line.strip()]
                    
                    if class_names:
                        self.log_update.emit(f"从classes.txt加载类名: {class_file}")
                        return class_names
                except Exception as e:
                    self.log_update.emit(f"从classes.txt读取类名失败: {str(e)}")
        
        # 如果上述方法都失败，尝试从标签文件中推断
        try:
            # 查找包含.txt文件的目录
            txt_dirs = []
            for root, dirs, files in os.walk(self.train_dir):
                if any(f.endswith('.txt') for f in files):
                    txt_dirs.append(root)
            
            if not txt_dirs:
                self.log_update.emit("未找到标签文件(.txt)")
                return []
            
            # 收集所有出现的类ID
            class_ids = set()
            for txt_dir in txt_dirs:
                for file in os.listdir(txt_dir):
                    if file.endswith('.txt'):
                        try:
                            with open(os.path.join(txt_dir, file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts and parts[0].isdigit():
                                        class_ids.add(int(parts[0]))
                        except Exception:
                            pass
            
            # 创建类名列表
            if class_ids:
                max_id = max(class_ids)
                class_names = [f"class{i}" for i in range(max_id + 1)]
                self.log_update.emit(f"从标签文件推断类名: 找到{len(class_names)}个类")
                return class_names
        
        except Exception as e:
            self.log_update.emit(f"从标签文件推断类名失败: {str(e)}")
        
        return []
    
    def _get_coco_class_names(self):
        """从COCO格式数据集中提取类名"""
        try:
            import json
            
            # Look for annotations file
            ann_file = None
            for file in os.listdir(self.train_dir):
                if file.endswith('.json') and ('annotations' in file or 'instances' in file):
                    ann_file = os.path.join(self.train_dir, file)
                    break
            
            if ann_file:
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Extract category names
                if 'categories' in coco_data:
                    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
                    class_names = [cat['name'] for cat in categories]
                    return class_names
        
        except Exception as e:
            self.log_update.emit(f"Error extracting COCO class names: {str(e)}")
        
        return []
    
    def _get_voc_class_names(self):
        """从VOC格式数据集中提取类名"""
        try:
            import xml.etree.ElementTree as ET
            
            # Get a list of all XML files
            xml_files = []
            for root, _, files in os.walk(self.train_dir):
                for file in files:
                    if file.endswith('.xml'):
                        xml_files.append(os.path.join(root, file))
            
            if not xml_files:
                self.log_update.emit("未找到VOC XML标注文件")
                return []
            
            # Extract unique class names from XML files
            class_names = set()
            for xml_file in xml_files[:10]:  # Only parse a few files for efficiency
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('.//object'):
                    name = obj.find('name').text
                    class_names.add(name)
            
            return sorted(list(class_names))
        
        except Exception as e:
            self.log_update.emit(f"Error extracting VOC class names: {str(e)}")
        
        return []
    
    def _find_common_parent(self, path1, path2):
        """找到两个路径的共同父目录"""
        path1 = os.path.abspath(path1)
        path2 = os.path.abspath(path2)
        
        # 将路径拆分为组件
        parts1 = path1.split(os.sep)
        parts2 = path2.split(os.sep)
        
        # 找到共同的前缀
        common_parts = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_parts.append(p1)
            else:
                break
        
        # 构建共同父路径
        if common_parts:
            common_path = os.sep.join(common_parts)
            # 在Windows上，确保包含驱动器号
            if os.name == 'nt' and not common_path.endswith(':'):
                common_path += os.sep
            return common_path
        
        # 如果没有共同部分，返回根目录
        return os.path.dirname(path1) 