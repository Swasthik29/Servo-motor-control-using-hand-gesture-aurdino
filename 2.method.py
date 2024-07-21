import cv2
import mediapipe as mp
import math
import pyfirmata
import numpy as np

# Port configuration for pyfirmata
my_port = 'COM9'
board = pyfirmata.Arduino(my_port)
iter8 = pyfirmata.util.Iterator(board)
iter8.start()

# Pin configuration for servo motor and LEDs
servo_pin = board.get_pin('d:9:s')
led1_pin = board.get_pin('d:2:o')  # Change to the appropriate pin number for LED 1
led2_pin = board.get_pin('d:3:o')  # Change to the appropriate pin number for LED 2
led3_pin = board.get_pin('d:4:o')  # Change to the appropriate pin number for LED 1
led4_pin = board.get_pin('d:5:o')  # Change to the appropriate pin number for LED 1
led5_pin = board.get_pin('d:6:o')  # Change to the appropriate pin number for LED 2



# Helper function to move the servo motor and control LEDs
def move_servo(angle):
    servo_pin.write(angle)
    if angle >= 0 and angle <= 36   :
        led1_pin.write(1)  # Turn on LED 1
    else:
        led1_pin.write(0)  # Turn off LED 1
    if angle >= 37 and angle <= 72 :
        led2_pin.write(1)  # Turn on LED 2
    else:
        led2_pin.write(0)  # Turn off LED 2
    if angle >= 73 and angle <= 108   :
        led3_pin.write(1)  # Turn on LED 1
    else:
        led3_pin.write(0)  # Turn off LED 1
    if angle >= 109 and angle <= 144 :
        led4_pin.write(1)  # Turn on LED 2
    else:
        led4_pin.write(0)  # Turn off LED 2
    if angle >= 145 and angle <= 180 :
        led5_pin.write(1)  # Turn on LED 2
    else:
        led5_pin.write(0)  # Turn off LED 2

# Function to draw dotted line in OpenCV
def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)
    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

# Initialize video capture
cap = cv2.VideoCapture(0)
distance = -19723086135

# MediaPipe hands configuration
with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        exit_x, exit_y = 700, 100
        exit_w, exit_h = 400, 100
        cv2.rectangle(image, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), (255, 0, 255), cv2.FILLED)
        cv2.putText(image, "Join your index and middle fingers to exit", (exit_x + 30, exit_y + 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    tips = [0, 4, 8, 12, 16, 20]
                    if id in tips:
                        cv2.circle(image, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
                drawline(image, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), (255, 255, 255), thickness=1, style='dotted', gap=10)
                angle = int(math.hypot(lmList[8][1] - lmList[4][1], lmList[8][2] - lmList[4][2]) / 2)
                print(angle)
                move_servo(angle)
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0)),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(201, 194, 2))
                )
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
