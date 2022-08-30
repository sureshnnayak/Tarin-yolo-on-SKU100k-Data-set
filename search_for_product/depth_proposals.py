import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

def draw_patch(img, patch, color=(255, 0, 0), thickness=4):
    '''
    Draws a patch on the image.
    '''
    cv2.rectangle(img, (patch[2], patch[0]), (patch[3], patch[1]), color=color, thickness=thickness)
    return img

once = True
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
    #plt.imshow(color_images_rgb)
    #plt.show()
    
    if once:
        once = False
        ax1 = plt.subplot(1,2,1)
        im1 = ax1.imshow(color_images_rgb)
        plt.ion()

    im1.set_data(color_images_rgb)
    plt.pause(0.00000001)
    print(color_images_rgb.shape[:2])
    

