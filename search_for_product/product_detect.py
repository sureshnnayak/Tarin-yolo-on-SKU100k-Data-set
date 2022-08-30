# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import pickle
import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

cwd = os.getcwd()
print(cwd)
print("Hellooo")
cwd = cwd + '/search_for_product'
print(cwd)
#os.chdir(cwd)
sys.path.insert(0, cwd)

from utils.augmentations import  letterbox

from models.common import DetectMultiBackend
#from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadImagesRealsense,LoadImagesCustom
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


from depth_proposals_autoencoder_coco import select_best_patch, draw_patch, get_similarity_score, yolov5_patch_to_our_patch

################## Initialization ###############################
product_index = 5 # 1 corresponds to Reese in the product databse as of 08/09
image_size = 128
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
transform = transforms.Compose( [transforms.ToPILImage(),
                               transforms.Resize((image_size, image_size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# Encoder
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5,5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.unconv1 = nn.ConvTranspose2d(6,3,kernel_size=(5,5))
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=(2,2))
        self.unmaxunpool2 = nn.MaxUnpool2d(kernel_size=(2,2))
        
        self.encoder1 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(6, 12,kernel_size=(5,5)),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(12, 16, kernel_size=(5,5)),
            nn.Tanh()
        )
        
    def encoder(self,x):
        x = self.conv1(x)
        x,indices1 = self.maxpool1(x)
        x = self.encoder1(x)
        x,indices2 = self.maxpool2(x)
        x = self.encoder2(x)
        return x

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        return x

# Compile Network
# autoencoder_model = autoencoder().to(device)
coc_color_pth = cwd + '/coco_color_128_16-16-25_autoencoder_statedict.pth'
#autoencoder_dict = torch.load('coco_color_128_16-16-25_autoencoder_statedict.pth')
autoencoder_dict = torch.load(coc_color_pth)
encoder_model = encoder().to(device)
encoder_dict = encoder_model.state_dict()

# 1. filter out unnecessary keys
autoencoder_dict = {k: v for k, v in autoencoder_dict.items() if k in encoder_dict}
# 2. overwrite entries in the existing state dict
encoder_dict.update(autoencoder_dict) 
# 3. load the new state dict
encoder_model.load_state_dict(autoencoder_dict)

resize = (image_size, image_size)
grid_shape = (5, 5)


with open(cwd + "/feature_list_encoder_coco.pickle", "rb") as f:
    feature_list = pickle.load(f)

##############################################################

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Yolo_model ():
    def __init__(self):
        print("Hello")
        self.weights=ROOT / 'best.pt'
        self.source = 'search_for_product/test_img2.png' # file/dir/URL/glob, 0 for webcam
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project= ROOT / 'runs/'  # save results to project/name
        self.name='detect'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        #   print(self.project)

        self.save_dir = self.project  / self.name
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size


@torch.no_grad()
def run(yolo_model, image):
    weights = yolo_model.weights
    source = yolo_model.source
    data = yolo_model.data
    imgsz = yolo_model.imgsz
    conf_thres = yolo_model.conf_thres
    iou_thres = yolo_model.iou_thres
    max_det = yolo_model.max_det
    device = yolo_model.device
    view_img = yolo_model.view_img
    save_txt = yolo_model.save_txt
    save_conf = yolo_model.save_conf
    save_crop = yolo_model.save_crop
    nosave = yolo_model.nosave
    classes = yolo_model.classes
    agnostic_nms = yolo_model.agnostic_nms
    augment = yolo_model.augment
    visualize = yolo_model.visualize
    update = yolo_model.update
    project = yolo_model.project
    name = yolo_model.name
    exist_ok = yolo_model.exist_ok
    line_thickness = yolo_model.line_thickness
    hide_labels = yolo_model.hide_labels
    hide_conf = yolo_model.hide_conf
    half = yolo_model.half
    dnn = yolo_model.dnn
    print(project)

    save_img = True #not nosave and not source.endswith('.txt')  # save inference images
    #save_img = not nosave and not source.endswith('.txt')  # save inference images
    '''once = True
    source = str(source)
    
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    '''
    # Directories

    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    '''
    save_dir = project  / name
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    '''
    save_dir = yolo_model.save_dir
    # Load model
    device = yolo_model.device
    model = yolo_model.model
    stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt
    imgsz = yolo_model.imgsz  # check image size

    bs = 1  # batch_size
    #vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

 
    path = source
    assert image is not None, f'Image Not Found {path}'
    s = f'image {1}/{1} {path}: '

    # Padded resize
    img = letterbox(image, imgsz, stride, pt)[0]

    # Convert
    im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)


    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

    #for path, im, image, vid_cap, s in dataset:
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference

    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3


    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, image)

    # Process predictions

    for i, det in enumerate(pred):  # per image
        seen += 1
        '''if webcam:  # batch_size >= 1
            p, im0, frame = path[i], image[i].copy(), dataset.count
            s += f'{i}: '
        else:
        '''
        #p, im0, frame = path, image.copy(), getattr(dataset, 'frame', 0)
        p, im0= path, image.copy()
        p = Path(p)  # to Path
        
        save_path = str(save_dir / p.name)  # im.jpg
        #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # print(det)
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            
            #print(det)
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            patches = []
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #with open(f'{txt_path}.txt', 'a') as f:
                    #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    patch = yolov5_patch_to_our_patch((xyxy[0].cpu().detach().numpy(), xyxy[1].cpu().detach().numpy(), xyxy[2].cpu().detach().numpy(), xyxy[3].cpu().detach().numpy()))
                    patches.append(patch)
                    annotator.box_label(xyxy, label, color=colors(c, True))
                
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            best_patch = select_best_patch(im0, patches, encoder_model, feature_list, product_index=product_index)
            best_patch_yolov5 = (best_patch[2], best_patch[0], best_patch[3], best_patch[1])
            annotator.box_label(best_patch_yolov5, "FOUND", color=colors(c+5, True))

        # Stream results
        im0 = annotator.result()
    
        #view_img = 1
        if view_img:
            #if once:
            #    once = False
            #    ax1 = plt.subplot(1,2,1)
            #    im1 = ax1.imshow(im0)
            #plt.ion()
            plt.imshow(im0)
            plt.show()
            
            #im1.set_data(im0) 
            #plt.pause(0.00000001)
            '''
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
            '''
        # Save results (image with detections)
        
        if save_img:
            #if dataset.mode == 'image':
            cv2.imwrite(save_path, cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            '''else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
            '''
        
    # Print time (inference-only)
    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    '''
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    '''
    return patches, best_patch

'''def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT , help='save results to project/name')
    parser.add_argument('--name', default='runs', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
'''

def main():

    #sys.path.insert(1, '/path/to/application/app/folder')
    check_requirements(exclude=('tensorboard', 'thop'))
    
    #to load the model
    yolo_model = Yolo_model()
    
    #image path
    source = 'search_for_product/test_img2.png'
    image = cv2.imread(source) 
    
    #finding the prodoct
    patches, best_patch = run(yolo_model, image)


if __name__ == "__main__":
    main()








