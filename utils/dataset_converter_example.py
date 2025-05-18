from dataset_converter import DatasetConverter

def main():
    # Initialize the converter
    converter = DatasetConverter()
    
    # Example 1: Convert COCO dataset
    coco_input_path = "path/to/coco/annotations.json"  # Path to COCO annotation file
    coco_output_path = "path/to/output/yolo_dataset"   # Path to save YOLO format dataset
    converter.convert_dataset(coco_input_path, coco_output_path, "coco")
    
    # Example 2: Convert VOC dataset
    voc_input_path = "path/to/voc/dataset"            # Path to VOC dataset directory
    voc_output_path = "path/to/output/yolo_dataset"   # Path to save YOLO format dataset
    converter.convert_dataset(voc_input_path, voc_output_path, "voc")

if __name__ == "__main__":
    main() 