import argparse
import sys
from pathlib import Path
import os
from search_for_product import product_detect
import cv2

from search_for_product.utils.general import  check_requirements


if __name__ == "__main__":
    import sys
    #dirname = sys.argv[1]
    #print(dirname)
    #sys.path.insert(1, '/path/to/application/app/folder')
    check_requirements(exclude=('tensorboard', 'thop'))
    
    #to load the model
    yolo_model = product_detect.Yolo_model()
    
    #image path
    source = 'search_for_product/test_img2.png'
    image = cv2.imread(source) 
    
    #finding the prodoct
    patches, best_patch = product_detect.run(yolo_model, image)
