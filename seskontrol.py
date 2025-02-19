# pip install cv2
pip install mediapipe
pip install numpy
pip install ctypes
pip install comtypes
pip install pycaw

import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Kamerayı başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

# Mediapipe el tespiti
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Pycaw ile ses kontrolü
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# Başlangıç değerleri
volbar = 400
volper = 0

while True:
    success, img = cap.read()
    if not success:
        continue  # Kamera görüntüsü alınamıyorsa döngüye devam et

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList:  # Boş listeyi kontrol et
        x1, y1 = lmList[4][1], lmList[4][2]  # Baş parmak
        x2, y2 = lmList[8][1], lmList[8][2]  # İşaret parmağı

        # Görsel işaretleme
        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Mesafe hesaplama
        length = hypot(x2 - x1, y2 - y1)

        # Ses seviyesini ayarla
        vol = np.interp(length, [30, 350], [volMin, volMax])
        volbar = np.interp(length, [30, 350], [400, 150])
        volper = np.interp(length, [30, 350], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        # Ses seviyesi çubuğu
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

    cv2.imshow('Image', img)

    # Çıkış tuşu olarak SPACE veya ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == 27:  # Space veya ESC tuşu
        break

cap.release()
cv2.destroyAllWindows()
