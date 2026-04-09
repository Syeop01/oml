import cv2
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
from jetbot import Robot  # Yahboom Jetbot specific library

# 1. Robot and Model Configuration
robot = Robot()
# GStreamer string for the IMX219 camera
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# 2. Load AI Model
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_model_xy.pth'))
device = torch.device('cuda')
model = model.to(device).eval().half() # Using half-precision for faster inference

# 3. Preprocessing and PID Control Variables
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

angle, angle_last = 0.0, 0.0
speed_gain_value = 0.18    # Base speed (0.0 to 1.0)
steering_gain_value = 0.03  # Proportional gain (P)
steering_dgain_value = 0.0  # Derivative gain (D)
steering_bias_value = 0.0   # Hardware bias compensation

def preprocess(image):
    """Normalizes and prepares the image for the model."""
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def executeModel(image):
    """Runs inference and controls the motors based on predicted coordinates."""
    global angle, angle_last

    # Model inference to get target (x, y)
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = -((0.5 - xy[1]) / 2.0)
   
    # PID control logic for steering angle
    angle = np.arctan2(x, y)
    pid = angle * steering_gain_value + (angle - angle_last) * steering_dgain_value
    angle_last = angle

    steering_value = pid + steering_bias_value
    
    # Yahboom Jetbot motor control (Values between 0.0 and 1.0)
    left_motor_value = max(min(speed_gain_value + steering_value, 1.0), 0.0)
    right_motor_value = max(min(speed_gain_value - steering_value, 1.0), 0.0)

    robot.left_motor.value = left_motor_value
    robot.right_motor.value = right_motor_value

def all_stop():
    """Standard stop command for the Yahboom library."""
    robot.stop()

def videoProcess(openpath):
    """Main loop for capturing video and running the control cycle."""
    cap = cv2.VideoCapture(openpath)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            executeModel(frame)
            cv2.imshow("Input", frame)
            
            # Press 'q' to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        print("Stop signal detected")
    finally:
        all_stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Yahboom Jetbot Autonomous Driving!")
    videoProcess(gst_str)