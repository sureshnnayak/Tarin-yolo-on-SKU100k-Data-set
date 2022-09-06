# Tarin-yolo-on-SKU100k-Data-set
Trainging YOLOv5 on SKU110K dataset 
1. Clone the YOLO repository from : https://github.com/ultralytics/yolov5
2. Download SKU110K dataset: https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd/edit
3. YOLOv5 model expects the annotation file to be in specific format. To convet the annotation from SKU110k to yolov5 format use run the convert_sku_to_yolo.pynb 
    refer this document for more details on yolo fpormat and training yolov5 on custom dataset : https://blog.paperspace.com/train-yolov5-custom-data/
5. Edit sku_dataset.yaml file.
6. train the model using the following command
 python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 100 --data road_sign_data.yaml --weights yolov5s.pt --workers 24 --name yolo_road_det
