import cv2
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
from torch2trt import TRTModule  # TensorRT support
from jetbot import Robot        # Yahboom specific library

# 1. Initialize Yahboom Robot
robot = Robot()

# GStreamer string for the IMX219 camera
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# 2. Load TensorRT Optimized Model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('best_model_xy_trt.pth'))

device = torch.device('cuda')
model = model_trt.to(device)
model = model_trt.eval()

# 3. Preprocessing and Control Variables
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

angle, angle_last = 0.0, 0.0
speed_gain_value = 0.18
steering_gain_value = 0.03
steering_dgain_value = 0.0
steering_bias_value = 0.0

def preprocess(image):
    """Prepares image for TensorRT model inference."""
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def executeModel(image):
    """Calculates steering and drives Yahboom motors."""
    global angle, angle_last

    # Inference using TensorRT module
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = 0.12  # Fixed vertical target for stability

    # Calculate steering using PID logic
    angle = np.arctan2(x, y)
    pid = angle * steering_gain_value + (angle - angle_last) * steering_dgain_value
    angle_last = angle

    steering_value = pid + steering_bias_value
    
    # Calculate motor values for Yahboom (Range: -1.0 to 1.0)
    left_value = max(min(speed_gain_value + steering_value, 1.0), -1.0)
    right_value = max(min(speed_gain_value - steering_value, 1.0), -1.0)

    # Apply values to Yahboom motors
    robot.left_motor.value = left_value
    robot.right_motor.value = right_value

def all_stop():
    """Emergency stop for Yahboom Jetbot."""
    robot.stop()

def Video(openpath):
    """Main video processing loop."""
    cap = cv2.VideoCapture(openpath)
    if not cap.isOpened():
        print("Failed to open camera. Aborting.")
        return

    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Resize to model input size
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            executeModel(frame_resized)
            
            cv2.imshow("Input", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        all_stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Yahboom TensorRT Autonomous Driving Ready")
    Video(gst_str)