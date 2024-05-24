import cv2
import time
from ultralytics import YOLO
from djitellopy import Tello

# Load the YOLOv8 model for human detection
model = YOLO("yolov8n.pt")

# PID controller class
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return output

# Connect to the drone
drone = Tello()
drone.connect()

# Get the video stream from the drone
drone.streamon()

# PID controller parameters (adjust as needed)q
pid_x = PIDController(Kp=0.2, Ki=0.001, Kd=0.1)
pid_y = PIDController(Kp=0.2, Ki=0.001, Kd=0.1)

# Take off to a height of 1.5 meters
drone.takeoff()
time.sleep(2)  # Wait for the drone to stabilize
drone.move_up(150)

start_time = time.time()

# Yaw until a person is detected
person_detected = False
while not person_detected:
    # Get the current frame
    frame = drone.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 detection
    results = model(frame)

    # Filter for human detections
    humans = results[0].boxes.boxes[results[0].boxes.cls == 0]

    if len(humans) > 0:
        person_detected = True
    else:
        drone.rotate_counter_clockwise(30)  # Rotate 30 degrees counter-clockwise

# Start tracking the person
while True:
    # Get the current frame
    frame = drone.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 detection
    results = model(frame)

    # Filter for human detections
    humans = results[0].boxes.boxes[results[0].boxes.cls == 0]

    if len(humans) > 0:
        # Get the centroid of the first detected human
        x, y, w, h = humans[0].xyxyn[0]
        centroid_x = int(x * frame.shape[1])
        centroid_y = int(y * frame.shape[0])

        # Calculate errors
        error_x = centroid_x - frame.shape[1] // 2
        error_y = centroid_y - frame.shape[0] // 2

        # Update PID controllers
        dt = time.time() - start_time
        output_x = pid_x.update(error_x, dt)
        output_y = pid_y.update(error_y, dt)

        # Send control commands to the drone
        drone.send_rc_control(int(output_x), int(output_y), 0, 0)

        # Draw the bounding box and centroid
        cv2.rectangle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), (int((x + w) * frame.shape[1]), int((y + h) * frame.shape[0])), (0, 255, 0), 2)
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    start_time = time.time()

# Clean up
drone.land()
drone.streamoff()
cv2.destroyAllWindows()