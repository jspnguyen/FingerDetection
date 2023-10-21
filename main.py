import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1, detectionCon=0.8)
capture = cv2.VideoCapture(0)

while True:
    status, frame = capture.read()
    if not status:
        break

    hands, frame = detector.findHands(frame, draw=False)
    num_digits = 0

    if hands:
        for hand in hands:
            lmlist = hand["lmList"] if "lmList" in hand else []
            if lmlist:
                fingerup = detector.fingersUp(hand)
                num_digits = fingerup.count(1)

    cv2.putText(frame, f'Fingers Held Up: {num_digits}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()