from simple_pid import PID
import cv2
from ultralytics import YOLO
import torchvision
import torch
import time

print("CUDA Available:", torch.cuda.is_available())
model = YOLO('yolov8s.pt')  # Assuming this is the correct path to the model

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #Change resolution here, change midpoint accordingly in pid_x, pid_roll
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# PID Controllers
pid_x = PID(0.1, 0.01, 0.05, setpoint=320)  # Horizontal movement (Yaw)
pid_z = PID(0.1, 0.01, 0.05, setpoint=0)  # Forward/backward movement
pid_roll = PID(0.1, 0.01, 0.05, setpoint=240)  # Roll control, setpoint at the center x of the frame

# Configure PID limits
pid_x.output_limits = (-10, 10)  # Horizontal speed limit
pid_z.output_limits = (-10, 10)  # Forward/backward speed limit
pid_roll.output_limits = (-10, 10)  # Roll speed limit

def send_uart_command(command):
    print("UART Command:", command)
    # Uncomment the next line to send commands over UART
    # ser.write(f'{command}\n'.encode())

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # resized_frame = cv2.resize(frame, (640, 640))
    results = model.predict(frame, conf=0.5)

    if results:
        result = results[0]
        if result.boxes and result.boxes.xyxy[0].numel() > 0:
            box = result.boxes.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height

            img_height, img_width, _ = frame.shape
            center_x, center_y = img_width // 2, img_height // 2

            # Update PID controllers
            x_adjust = pid_x(center_x - x_center)
            z_adjust = pid_z(center_y - y_center)
            roll_adjust = pid_roll(x_center)

            # Convert PID output to commands
            if x_adjust < 0:
                send_uart_command('YAW LEFT')
            elif x_adjust > 0:
                send_uart_command('YAW RIGHT')

            if z_adjust < 0:
                send_uart_command('FORWARD')
            elif z_adjust > 0:
                send_uart_command('BACKWARD')

            if roll_adjust < 0:
                send_uart_command('ROLL LEFT')
            elif roll_adjust > 0:
                send_uart_command('ROLL RIGHT')

            # Drawing the bounding box and the arrow for visualization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (center_x, center_y), (int(x_center), int(y_center)), (0, 255, 255), 2, tipLength=0.4)
            
        else:
            print("No boxes detected")
    
        # Display FPS information
        cv2.putText(frame, f"FPS: {fps:.2f}", (img_width - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('YOLO Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Uncomment the next line if you're using UART
# ser.close()
