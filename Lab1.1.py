import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

canvas = None
brush_thickness = 5
prev_x, prev_y = None, None
brush_color = (0, 0, 255)

def fingers_up(hand_landmarks, h):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    for tip in tips:
        y_tip = int(hand_landmarks.landmark[tip].y * h)
        y_base = int(hand_landmarks.landmark[tip - 2].y * h)
        fingers.append(1 if y_tip < y_base else 0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_info = list(zip(results.multi_handedness, results.multi_hand_landmarks))

        for hand_type, hand_landmarks in hand_info:
            label = hand_type.classification[0].label
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)

            if label == "Right":
                fingers = fingers_up(hand_landmarks, h)

                if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                    brush_color = (0, 0, 255)
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                    brush_color = (255, 0, 0)
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    brush_color = (0, 255, 0)

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x_index, y_index
                cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), brush_color, brush_thickness)
                prev_x, prev_y = x_index, y_index

            if label == "Left":

                distance = math.hypot(x_index - x_thumb, y_index - y_thumb)
                brush_thickness = int(np.interp(distance, [20, 200], [1, 50]))

                center_x = (x_index + x_thumb) // 2
                center_y = (y_index + y_thumb) // 2
                cv2.circle(frame, (center_x, center_y), brush_thickness, (255, 0, 0), -1)

    else:
        prev_x, prev_y = None, None

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("AirPaint", cv2.resize(frame, (1000, 700)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
