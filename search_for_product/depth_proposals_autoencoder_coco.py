import math
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from sklearn.mixture import GaussianMixture
from sympy import product
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms


################################################################

def get_similarity_score(patch1, patch2, type='euclidean'):
    '''
    Returns the similarity score between two patches.
    '''
    if type == 'cosine':
        return np.dot(patch1, patch2)/(np.linalg.norm(patch1)*np.linalg.norm(patch2))
    elif type == 'euclidean':
        return np.linalg.norm(patch1/np.linalg.norm(patch1)-patch2/np.linalg.norm(patch2))
    elif type == 'manhattan':
        return np.sum(np.abs(patch1-patch2))

def draw_patch(img, patch, color=(255, 0, 0), thickness=4):
    '''
    Draws a patch on the image.
    '''
    cv2.rectangle(img, (patch[2], patch[0]), (patch[3], patch[1]), color=color, thickness=thickness)
    return img

def select_best_patch(color_image, patches, model, feature_list, product_index):
    '''
    Selects the best patch from the list of patches.
    '''
    best_patch = None
    best_score = -1
    patch_images = []
    for i, patch in enumerate(patches):
        patch_image = color_image[patch[0]:patch[1], patch[2]:patch[3]]
        patch_image = transform(patch_image)
        # patch_image_bgr = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('patch: ' + str(i), patch_image_bgr)
        patch_images.append(patch_image)
    patch_images = torch.stack(patch_images)
    print(patch_images.shape)
    
    patch_images = Variable(patch_images).to(device)
    feature_output = model(patch_images)
    print(feature_output.shape)
    # feature_output = feature_output.cpu().squeeze().detach().numpy().flatten()
    print("Frame processed: ", feature_output.shape)
    feature_vector = feature_output
    target_product = feature_list[product_index]
    for i in range(len(patches)):
        patch_feature = feature_vector[i].cpu().detach().numpy().flatten()

        print(patch_feature.shape), print(target_product.shape)
        score = get_similarity_score(patch_feature, target_product, type='cosine')
        if score > best_score:
            best_patch = i
            best_score = score
    print("Best score: ", best_score)
    
    return patches[best_patch]

def yolov5_patch_to_our_patch(patch):
    '''
    Converts the patch from YOLOv5 format to our format.
    '''
    return (int(patch[1]), int(patch[3]), int(patch[0]), int(patch[2]))

def get_square_patches(depth_image, grid_shape, foreground_mask):
    '''
    Returns a list of square patches of size (patch_size, patch_size) from the depth image.
    '''
    h, w = depth_image.shape
    patches = []
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    for i in range(rows):
        for j in range(cols):
            x = int(dx*j)
            y = int(dy*i)
            patch = (y,y+int(dy), x, x+int(dx))
            if True or np.sum(foreground_mask[y:y+int(dy), x:x+int(dx)])/int(dx)*int(dy) > 0.3:
                patches.append(patch)
    for i in range(rows-1):
        for j in range(cols):
            x = int(dx*j)
            y = int(dy*i)
            patch = (y,y+int(2*dy), x, x+int(dx))
            if True or np.sum(foreground_mask[y:y+int(dy), x:x+int(dx)])/int(dx)*int(dy) > 0.3:
                patches.append(patch)
    return patches

def get_foreground_mask(depth_image):
    '''
    Returns a mask of the foreground (non-background) pixels in the depth image.
    '''
    gm = GaussianMixture(n_components=2, random_state=0).fit(depth_image)
    foreground_mask = gm.predict(depth_image)
    return foreground_mask

def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    '''
    Draws a grid on the image.
    Src: https://stackoverflow.com/a/69097578/1641628
    '''
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

image_size = 128
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
transform = transforms.Compose( [transforms.ToPILImage(),
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

if __name__ == "__main__":

    ################## Initialization ###############################
    product_index = 3
    
    
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
    autoencoder_dict = torch.load('coco_color_128_16-16-25_autoencoder.pth')
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


    with open("feature_list_encoder_coco.pickle", "rb") as f:
        feature_list = pickle.load(f)



    ################## REALSENSE STUFF ##############################

    REALSENSE_FOCAL_LENGTH = 1.93  # millimeters
    REALSENSE_SENSOR_HEIGHT_MM = 2.454  # millimeters
    REALSENSE_SENSOR_WIDTH_MM = 3.896  # millimeters
    REALSENSE_REZ_HEIGHT = 800  # pixels
    REALSENSE_REZ_WIDTH = 1280  # pixels
    REALSENSE_FX = 628.071 # D455
    REALSENSE_PPX = 637.01 # D455
    REALSENSE_FX = 383.0088195800781 # D455
    REALSENSE_PPX = 320.8406066894531 # D455
    REALSENSE_PPY = 238.125 #D455
    REALSENSE_FY = 383.0088195800781 #D455

    # ====== Realsense ======
    realsense_ctx = rs.context()
    connected_devices = [] # List of serial numbers for present cameras
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        print(f"{detected_camera}")
        connected_devices.append(detected_camera)
    #device = connected_devices[0] # use this when we are only using one camera
    device_rgb = connected_devices[0] if len(connected_devices) > 0 else '115422250228' #Hardcoding the device Number 
    pipeline_rgb = rs.pipeline()
    config_rgb = rs.config()
    background_removed_color = 153 # Grey

    # ====== Enable Streams ======
    config_rgb.enable_device(device_rgb)

    # # For worse FPS, but better resolution:
    # stream_res_x = 1280
    # stream_res_y = 720
    # # For better FPS. but worse resolution:
    stream_res_x = 640
    stream_res_y = 480

    stream_fps = 30

    config_rgb.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
    config_rgb.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
    profile = pipeline_rgb.start(config_rgb)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # ====== Get depth Scale ======
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth:")
    print(depth_scale)
    print(f"\tDepth Scale for Camera SN {device_rgb} is: {depth_scale}")

    # ====== Get and process images ====== 
    print(f"Starting to capture images on SN: {device_rgb}")

    #################################################################
    while True:
        # Get and align frames
        frames = pipeline_rgb.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
            
        if not aligned_depth_frame or not color_frame:
            continue

        # Process images
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image,1)
        color_image = np.asanyarray(color_frame.get_data())

        dimensions = color_image.shape
        
        #images = color_image
        # color_image = cv2.flip(color_image,1)
        images = color_image

        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # color_images_rgb = color_image

        # img = cv2.imread("shelf.png")
        img_gray = cv2.cvtColor(color_images_rgb, cv2.COLOR_BGR2GRAY)
        cdst = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        # Threshold 
        _, img_thr = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        # Edge detection
        dst = cv2.Canny(img_gray, 50, 200, None, 3)

        grided_image = draw_grid(color_images_rgb, grid_shape)
        depth_image_flipped_clustering = depth_image_flipped.reshape(-1, 1)
        foreground_mask = get_foreground_mask(depth_image_flipped_clustering)
        foreground_mask = foreground_mask.reshape(depth_image_flipped.shape)
        # foreground_coords = np.nonzero(foreground_mask == 0)
        # color_images_without_foreground = np.copy(color_images_rgb)
        # color_images_without_foreground[foreground_coords[0], foreground_coords[1], :] = (0, 0, 0)
        # background = cv2.bitwise_and(color_images_rgb, 255 - mask)
        patches = get_square_patches(depth_image_flipped, grid_shape, foreground_mask)
        best_patch = select_best_patch(color_images_rgb, patches, encoder_model, feature_list, product_index=product_index)
        best_patch_image = cv2.cvtColor(color_images_rgb, cv2.COLOR_RGB2BGR)
        if best_patch is not None:
            best_patch_image = draw_patch(color_images_rgb, best_patch)
            best_patch_image = cv2.cvtColor(best_patch_image, cv2.COLOR_RGB2BGR)

        



        # cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        # cv2.imshow("Source", color_images_rgb)    
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        img_thr = cv2.cvtColor(img_thr, cv2.COLOR_GRAY2BGR)
        display_image = np.concatenate((img_thr, cdstP), axis=1)
        cv2.imshow("Depth Proposals", best_patch_image)
        key = cv2.waitKey(1)

        # break

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {device_rgb}")
            break