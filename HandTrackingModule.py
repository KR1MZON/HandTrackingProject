import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print (results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def finpos(self, img, handnum=0, draw=True):
        landmarklist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handnum]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                landmarklist.append([id, cx, cy])
                if draw:
                    if (id == 0):
                        cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        return landmarklist


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detecor = handDetector()
    while True:
        success, img = cap.read()
        img = detecor.findhands(img)
        landmarklist = detecor.finpos(img)
        if len(landmarklist) != 0:
            print(landmarklist[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
