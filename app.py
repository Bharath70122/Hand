import cv2
import mediapipe as mp
import webbrowser
import time

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

# ---------------- CONTROL ----------------
last_trigger_time = 0
cooldown = 3

# ---------------- FINGER COUNT ----------------
def count_fingers(hand_landmarks, hand_label):
    tips = [8, 12, 16, 20]
    fingers = []

    # ---------------- THUMB ----------------
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    if hand_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if thumb_tip.x > thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)

    # ---------------- OTHER FINGERS ----------------
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for idx, handLms in enumerate(result.multi_hand_landmarks):

            draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            hand_label = result.multi_handedness[idx].classification[0].label

            fingers = count_fingers(handLms, hand_label)
            finger_count = sum(fingers)

            print(hand_label, "FINGERS:", fingers, "SUM:", finger_count)

            current_time = time.time()

            # ---------------- GESTURES ----------------

            # 5 → YouTube
            if finger_count == 5:
                if current_time - last_trigger_time > cooldown:
                    print("🚀 OPENING YOUTUBE")
                    webbrowser.open_new("https://www.youtube.com")
                    last_trigger_time = current_time

            # 4 → Instagram
            elif finger_count == 4:
                if current_time - last_trigger_time > cooldown:
                    print("📸 OPENING INSTAGRAM")
                    webbrowser.open_new("https://www.instagram.com")
                    last_trigger_time = current_time

            # 3 → GitHub
            elif finger_count == 3:
                if current_time - last_trigger_time > cooldown:
                    print("💻 OPENING GITHUB")
                    webbrowser.open_new("https://github.com")
                    last_trigger_time = current_time

            # 2 → Facebook
            elif finger_count == 2:
                if current_time - last_trigger_time > cooldown:
                    print("📘 OPENING FACEBOOK")
                    webbrowser.open_new("https://www.facebook.com")
                    last_trigger_time = current_time

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()