import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Union
import numpy as np
from tqdm import tqdm

class DatasetConverter:
    def __init__(self):
        self.supported_formats = ['coco', 'voc']
        
    def convert_dataset(self, input_path: str, output_path: str, format_type: str, mode: str = 'overall', 
                       train_images_dir: str = None, train_labels_dir: str = None,
                       val_images_dir: str = None, val_labels_dir: str = None,
                       val_ratio: float = 0.2) -> bool:
        """
        Convert dataset from COCO or VOC format to YOLO format
        
        Args:
            input_path (str): Path to input dataset
            output_path (str): Path to save converted dataset
            format_type (str): Format type ('coco' or 'voc')
            mode (str): Conversion mode ('overall' or 'split')
            train_images_dir (str): Path to training images directory (for split mode)
            train_labels_dir (str): Path to training labels directory (for split mode)
            val_images_dir (str): Path to validation images directory (for split mode)
            val_labels_dir (str): Path to validation labels directory (for split mode)
            val_ratio (float): Ratio of validation set (for overall mode)
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        if format_type.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format type. Supported formats are: {self.supported_formats}")
            
        if mode not in ['overall', 'split']:
            raise ValueError("Mode must be either 'overall' or 'split'")
            
        if mode == 'split' and (not train_images_dir or not train_labels_dir or not val_images_dir or not val_labels_dir):
            raise ValueError("In split mode, all train/val directories must be specified")
            
        if format_type.lower() == 'coco':
            return self._convert_coco_to_yolo(input_path, output_path, mode, 
                                            train_images_dir, train_labels_dir,
                                            val_images_dir, val_labels_dir,
                                            val_ratio)
        else:  # VOC format
            return self._convert_voc_to_yolo(input_path, output_path, mode,
                                           train_images_dir, train_labels_dir,
                                           val_images_dir, val_labels_dir,
                                           val_ratio)
    
    def _validate_coco_format(self, data: dict) -> bool:
        """
        Validate if the data follows COCO format
        
        Args:
            data (dict): The loaded JSON data
            
        Returns:
            bool: True if valid COCO format, False otherwise
        """
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in data:
                print(f"Error: Missing required field '{field}' in COCO format")
                return False
        
        if not isinstance(data['images'], list):
            print("Error: 'images' field must be a list")
            return False
            
        if not isinstance(data['annotations'], list):
            print("Error: 'annotations' field must be a list")
            return False
            
        if not isinstance(data['categories'], list):
            print("Error: 'categories' field must be a list")
            return False
            
        if not data['categories']:
            print("Error: No categories defined in the dataset")
            return False
            
        return True

    def _convert_coco_to_yolo(self, input_path: str, output_path: str, mode: str,
                             train_images_dir: str, train_labels_dir: str,
                             val_images_dir: str, val_labels_dir: str,
                             val_ratio: float) -> bool:
        """
        Convert COCO format dataset to YOLO format
        
        Args:
            input_path (str): Path to COCO dataset
            output_path (str): Path to save YOLO format dataset
            mode (str): Conversion mode ('overall' or 'split')
            train_images_dir (str): Path to training images directory (for split mode)
            train_labels_dir (str): Path to training labels directory (for split mode)
            val_images_dir (str): Path to validation images directory (for split mode)
            val_labels_dir (str): Path to validation labels directory (for split mode)
            val_ratio (float): Ratio of validation set (for overall mode)
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Create output directories
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
            
            # Check if input path is a directory or file
            if os.path.isdir(input_path):
                # Look for annotation files in the directory
                json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
                if not json_files:
                    raise FileNotFoundError(f"No JSON annotation files found in {input_path}")
                input_path = os.path.join(input_path, json_files[0])
                print(f"Using annotation file: {input_path}")
            
            # Check if the annotation file exists and is readable
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Annotation file not found: {input_path}")
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
            except PermissionError:
                raise PermissionError(f"Permission denied when reading annotation file: {input_path}. Please check file permissions.")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in annotation file: {input_path}")
            
            # Validate COCO format
            if not self._validate_coco_format(coco_data):
                raise ValueError(f"Invalid COCO format in annotation file: {input_path}")
            
            # Create category mapping
            categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
            print(f"Found {len(categories)} categories: {list(categories.values())}")
            
            if mode == 'overall':
                # Get all images and shuffle for random split
                images = coco_data['images']
                if not images:
                    raise ValueError("No images found in the annotation file")
                
                print(f"Found {len(images)} images in the dataset")
                np.random.shuffle(images)
                
                # Calculate split index
                split_idx = int(len(images) * (1 - val_ratio))
                train_images = images[:split_idx]
                val_images = images[split_idx:]
                
                print(f"Processing {len(train_images)} training images and {len(val_images)} validation images")
                
                # Process training images
                for img in tqdm(train_images, desc="Converting training set"):
                    self._process_coco_image(img, coco_data, train_images_dir, output_path, 'train')
                
                # Process validation images
                for img in tqdm(val_images, desc="Converting validation set"):
                    self._process_coco_image(img, coco_data, train_images_dir, output_path, 'val')
            else:  # split mode
                # Process training set
                train_ann_file = os.path.join(train_labels_dir, 'instances_train.json')
                if os.path.exists(train_ann_file):
                    with open(train_ann_file, 'r', encoding='utf-8') as f:
                        train_data = json.load(f)
                    if not self._validate_coco_format(train_data):
                        raise ValueError(f"Invalid COCO format in training annotation file: {train_ann_file}")
                    for img in tqdm(train_data['images'], desc="Converting training set"):
                        self._process_coco_image(img, train_data, train_images_dir, output_path, 'train')
                
                # Process validation set
                val_ann_file = os.path.join(val_labels_dir, 'instances_val.json')
                if os.path.exists(val_ann_file):
                    with open(val_ann_file, 'r', encoding='utf-8') as f:
                        val_data = json.load(f)
                    if not self._validate_coco_format(val_data):
                        raise ValueError(f"Invalid COCO format in validation annotation file: {val_ann_file}")
                    for img in tqdm(val_data['images'], desc="Converting validation set"):
                        self._process_coco_image(img, val_data, val_images_dir, output_path, 'val')
            
            # Create dataset.yaml
            yaml_content = {
                'path': output_path,
                'train': 'images/train',
                'val': 'images/val',
                'names': {i: name for i, name in enumerate(categories.values())}
            }
            
            with open(os.path.join(output_path, 'dataset.yaml'), 'w') as f:
                import yaml
                yaml.dump(yaml_content, f)
            
            return True
            
        except Exception as e:
            print(f"Error converting COCO dataset: {str(e)}")
            return False
    
    def _process_coco_image(self, img, coco_data, images_dir, output_path, split_type):
        """Helper method to process a single COCO image"""
        img_id = img['id']
        img_file = img['file_name']
        img_width = img['width']
        img_height = img['height']
        
        # Try to find the image file
        possible_paths = [
            # Check in the images directory
            os.path.join(images_dir, img_file),
            os.path.join(images_dir, os.path.basename(img_file)),
            # Check in common subdirectories
            os.path.join(images_dir, 'images', img_file),
            os.path.join(images_dir, 'images', os.path.basename(img_file)),
            os.path.join(images_dir, 'train', img_file),
            os.path.join(images_dir, 'train', os.path.basename(img_file)),
            os.path.join(images_dir, 'val', img_file),
            os.path.join(images_dir, 'val', os.path.basename(img_file)),
            # Check in parent directories
            os.path.join(os.path.dirname(images_dir), 'images', img_file),
            os.path.join(os.path.dirname(images_dir), 'images', os.path.basename(img_file)),
            os.path.join(os.path.dirname(images_dir), 'train', img_file),
            os.path.join(os.path.dirname(images_dir), 'train', os.path.basename(img_file)),
            os.path.join(os.path.dirname(images_dir), 'val', img_file),
            os.path.join(os.path.dirname(images_dir), 'val', os.path.basename(img_file)),
            # Check in the same directory as the annotation file
            os.path.join(os.path.dirname(images_dir), img_file),
            os.path.join(os.path.dirname(images_dir), os.path.basename(img_file))
        ]
        
        # Try each possible path
        src_img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                src_img_path = path
                break
        
        if src_img_path is None:
            print(f"Warning: Image file not found: {img_file}")
            print(f"Tried paths: {possible_paths}")
            print(f"Please ensure the image file exists in one of these locations or update the images_dir parameter.")
            return
        
        # Copy image file
        dst_img_path = os.path.join(output_path, 'images', split_type, os.path.basename(img_file))
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        
        # Copy image file
        try:
            shutil.copy2(src_img_path, dst_img_path)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {src_img_path}")
            return
        
        # Create YOLO format labels
        yolo_labels = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                # Convert bbox to YOLO format (x_center, y_center, width, height)
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # Get class id
                class_id = ann['category_id'] - 1  # YOLO uses 0-based indexing
                
                yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        # Save labels
        label_file = os.path.join(output_path, 'labels', split_type,
                                os.path.splitext(os.path.basename(img_file))[0] + '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))
    
    def _convert_voc_to_yolo(self, input_path: str, output_path: str, mode: str,
                            train_images_dir: str, train_labels_dir: str,
                            val_images_dir: str, val_labels_dir: str,
                            val_ratio: float) -> bool:
        """
        Convert VOC format dataset to YOLO format
        
        Args:
            input_path (str): Path to VOC dataset
            output_path (str): Path to save YOLO format dataset
            mode (str): Conversion mode ('overall' or 'split')
            train_images_dir (str): Path to training images directory (for split mode)
            train_labels_dir (str): Path to training labels directory (for split mode)
            val_images_dir (str): Path to validation images directory (for split mode)
            val_labels_dir (str): Path to validation labels directory (for split mode)
            val_ratio (float): Ratio of validation set (for overall mode)
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Create output directories
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
            
            if mode == 'overall':
                # Get all XML files
                xml_files = []
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.endswith('.xml'):
                            xml_files.append(os.path.join(root, file))
                
                # Shuffle files for random split
                np.random.shuffle(xml_files)
                
                # Calculate split index
                split_idx = int(len(xml_files) * (1 - val_ratio))
                train_files = xml_files[:split_idx]
                val_files = xml_files[split_idx:]
                
                # Process training files
                class_names = set()
                for xml_file in tqdm(train_files, desc="Converting training set"):
                    class_names.update(self._process_voc_file(xml_file, output_path, 'train', train_images_dir))
                
                # Process validation files
                for xml_file in tqdm(val_files, desc="Converting validation set"):
                    class_names.update(self._process_voc_file(xml_file, output_path, 'val', train_images_dir))
            else:  # split mode
                # Process training set
                train_xml_files = []
                for root, _, files in os.walk(train_labels_dir):
                    for file in files:
                        if file.endswith('.xml'):
                            train_xml_files.append(os.path.join(root, file))
                
                # Process validation set
                val_xml_files = []
                for root, _, files in os.walk(val_labels_dir):
                    for file in files:
                        if file.endswith('.xml'):
                            val_xml_files.append(os.path.join(root, file))
                
                # Process training files
                class_names = set()
                for xml_file in tqdm(train_xml_files, desc="Converting training set"):
                    class_names.update(self._process_voc_file(xml_file, output_path, 'train', train_images_dir))
                
                # Process validation files
                for xml_file in tqdm(val_xml_files, desc="Converting validation set"):
                    class_names.update(self._process_voc_file(xml_file, output_path, 'val', val_images_dir))
            
            # Create dataset.yaml
            yaml_content = {
                'path': output_path,
                'train': 'images/train',
                'val': 'images/val',
                'names': {i: name for i, name in enumerate(sorted(list(class_names)))}
            }
            
            with open(os.path.join(output_path, 'dataset.yaml'), 'w') as f:
                import yaml
                yaml.dump(yaml_content, f)
            
            return True
            
        except Exception as e:
            print(f"Error converting VOC dataset: {str(e)}")
            return False
    
    def _process_voc_file(self, xml_file, output_path, split_type, images_dir=None):
        """Helper method to process a single VOC XML file"""
        class_names = set()
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image filename
            filename_elem = root.find('filename')
            if filename_elem is None:
                print(f"Warning: No filename element found in {xml_file}")
                return class_names
            img_filename = filename_elem.text
            if img_filename is None:
                print(f"Warning: Empty filename in {xml_file}")
                return class_names
            
            # Try multiple possible locations for the image
            possible_paths = []
            if images_dir:
                possible_paths.append(os.path.join(images_dir, img_filename))
            else:
                possible_paths.extend([
                    os.path.join(os.path.dirname(xml_file), 'JPEGImages', img_filename),
                    os.path.join(os.path.dirname(os.path.dirname(xml_file)), 'JPEGImages', img_filename),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(xml_file))), 'JPEGImages', img_filename),
                    os.path.join(os.path.dirname(xml_file), img_filename),
                    os.path.join(os.path.dirname(os.path.dirname(xml_file)), img_filename),
                ])
            
            # Try each possible path
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break
            
            if img_path is None:
                print(f"Warning: Image file not found: {img_filename}")
                return class_names
            
            # Get image dimensions
            size = root.find('size')
            if size is None:
                print(f"Warning: No size element found in {xml_file}")
                return class_names
                
            width_elem = size.find('width')
            height_elem = size.find('height')
            if width_elem is None or height_elem is None:
                print(f"Warning: Missing width or height in {xml_file}")
                return class_names
                
            try:
                img_width = int(width_elem.text)
                img_height = int(height_elem.text)
            except (ValueError, TypeError):
                print(f"Warning: Invalid width or height values in {xml_file}")
                return class_names
            
            # Copy image file
            dst_img_path = os.path.join(output_path, 'images', split_type, os.path.basename(img_filename))
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            
            # Copy image file
            try:
                shutil.copy2(img_path, dst_img_path)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {img_path}")
                return class_names
            
            # Process annotations
            yolo_labels = []
            for obj in root.findall('object'):
                # Try both 'n' and 'name' tags for class name
                name_elem = obj.find('n')
                if name_elem is None:
                    name_elem = obj.find('name')
                    if name_elem is None:
                        print(f"Warning: No name element found in object in {xml_file}")
                        continue
                        
                class_name = name_elem.text
                if class_name is None:
                    print(f"Warning: Empty class name in {xml_file}")
                    continue
                    
                class_names.add(class_name)
                
                bbox = obj.find('bndbox')
                if bbox is None:
                    print(f"Warning: No bndbox element found in {xml_file}")
                    continue
                    
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                except (ValueError, TypeError, AttributeError):
                    print(f"Warning: Invalid bbox values in {xml_file}")
                    continue
                
                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_labels.append(f"{list(class_names).index(class_name)} {x_center} {y_center} {width} {height}")
            
            # Save labels
            label_file = os.path.join(output_path, 'labels', split_type,
                                    os.path.splitext(os.path.basename(img_filename))[0] + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
        except ET.ParseError as e:
            print(f"Warning: Error parsing XML file {xml_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error processing {xml_file}: {str(e)}")
            
        return class_names 