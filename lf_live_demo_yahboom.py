import cv2
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image
import time
from jetbotmini import Robot

import ipywidgets
from IPython.display import display

# 1. Initialize Robot and Camera Pipeline
robot = Robot()
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# 2. Load Standard PyTorch Model
print("Loading Standard PyTorch Model...")
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_model_xy.pth')) 

device = torch.device('cuda')
model = model.to(device).eval().half()
print("Model Loading Complete!")

# Data Normalization Parameters
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

# 3. Tuned PID & Motor Parameters
angle, angle_last = 0.0, 0.0
speed_gain_value = 0.35      # Base speed (Overcomes static friction)
steering_gain_value = 0.15   # Steering sensitivity (Increased for sharp cornering)
steering_dgain_value = 0.05  # Derivative gain (Prevents overshoot during turns)
steering_bias_value = 0.0    # Mechanical bias adjustment

# Create Jupyter Widget for Display
image_widget = ipywidgets.Image(format='jpeg', width=224, height=224)
display(image_widget)

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def executeModel(image):
    global angle, angle_last

    # Inference
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = 0.12 # Look-ahead distance

    # Calculate Steering Angle and PID
    angle = np.arctan2(x, y)
    pid = angle * steering_gain_value + (angle - angle_last) * steering_dgain_value
    angle_last = angle

    steering_value = pid + steering_bias_value
    
    # Calculate Motor Speeds
    left_v = max(min(speed_gain_value + steering_value, 1.0), -1.0)
    right_v = max(min(speed_gain_value - steering_value, 1.0), -1.0)

    # Apply Motor Speeds
    robot.left_motor.value = left_v
    robot.right_motor.value = right_v

def all_stop():
    robot.stop()

def Video(openpath):
    cap = cv2.VideoCapture(openpath)
    if not cap.isOpened():
        print("Camera Error: Please restart the kernel.")
        return

    print("Standard Auto-Driving Started!")
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Execute Model & Control Motors
            executeModel(frame_resized)
            
            # Update Display every 15 frames to prevent browser freeze
            if frame_count % 15 == 0:
                _, jpeg = cv2.imencode('.jpg', frame_resized)
                image_widget.value = jpeg.tobytes()

            frame_count += 1
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopped by User.")
    finally:
        all_stop()
        cap.release()

# Run the Video loop
Video(gst_str)