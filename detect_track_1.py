from ultralytics import YOLO
import torchvision
import torch
import cv2
# import serial  # Uncomment if using UART

print("CUDA Available:", torch.cuda.is_available())
model = YOLO('yolov8s.pt')

# Uncomment if using UART
# ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def send_uart_command(command):
    print("UART Command:", command)
    # ser.write(f'{command}\n'.encode())

def resize_image(frame, size):
    return cv2.resize(frame, (size, size))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    resized_frame = resize_image(frame, 640)
    results = model.predict(resized_frame, conf=0.5)

    if results:
        result = results[0]
        if result.boxes and result.boxes.xyxy[0].numel() > 0:
            box = result.boxes.xyxy[0].cpu().tolist()  # Assuming a single box
            x1, y1, x2, y2 = box
            print(f"Box coordinates: {x1}, {y1}, {x2}, {y2}")

            label = 'Detected Object'  # Placeholder label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height

            img_height, img_width, _ = frame.shape
            center_x, center_y = img_width // 2, img_height // 2

            if x_center < center_x - 10:
                send_uart_command('LEFT')
            elif x_center > center_x + 10:
                send_uart_command('RIGHT')
            if y_center < center_y - 10:
                send_uart_command('UP')
            elif y_center > center_y + 10:
                send_uart_command('DOWN')

            if area < 50000:
                send_uart_command('FORWARD')
            elif area > 150000:
                send_uart_command('BACKWARD')
        else:
            print("No boxes detected")

    cv2.imshow('YOLO Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# ser.close()
