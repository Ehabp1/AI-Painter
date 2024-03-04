import numpy as np
import cv2 as cv
import mediapipe as mp
import os



def mode(arr):
    fingers = []
    if arr[8][2] < arr[6][2]:
        fingers.append(1)
    else:
        fingers.append(0)
    if arr[12][2] < arr[10][2]:
        fingers.append(1)
    else:
        fingers.append(0)
    return fingers






video = cv.VideoCapture(0)
video.set(3,1280)
video.set(4,720)


media = mp.solutions.hands
hand = media.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils
# load header images
img_path =os.path.join(os.getcwd(),'header')
print(os.listdir(img_path))
header = []
col=(255,255,255)
for img in os.listdir(img_path):
    cur_path = os.path.join(img_path,img)
    im = cv.imread(cur_path)
    im = cv.resize(im,(1280,100))
    header.append(im)
print(len(header))
xp,yp =(0,0)
can = np.zeros((720,1280,3),dtype=np.uint8)
cv.imshow('im',can)
while True:
    success, frame = video.read()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_frame = cv.flip(rgb_frame,1)
    result = hand.process(rgb_frame)
    loc = []
    if result.multi_hand_landmarks:
        for det in result.multi_hand_landmarks:
            for id, lm in enumerate(det.landmark):
                h, w, c = rgb_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                loc.append([id,cx,cy])

            draw.draw_landmarks(rgb_frame, det, media.HAND_CONNECTIONS)
        up = mode(loc)
        x1,y1 = loc[8][1],loc[8][2]
        x2, y2 = loc[12][1], loc[12][2]

        #selection mode
        if up[0] and up[1]:
            print('selection')
            if 0 < x1 < 300:
                rgb_frame[:100, :1500] = header[0]
                col = (255, 0, 255)
                xp, yp = (0, 0)
            elif 300 < x1 < 740:
                rgb_frame[:100, :1500] = header[1]
                col = (255, 100, 0)
                xp, yp = (0, 0)
            elif 740 < x1 < 1000:
                rgb_frame[:100, :1500] = header[2]
                col = (0, 255, 0)
                xp, yp = (0, 0)
            elif x1 > 1000:
                rgb_frame[:100, :1500] = header[3]
                col = (0, 0, 0)
            cv.rectangle(rgb_frame, (x1, y1), (x2, y2), col, 2)
            xp, yp = (0, 0)

        #drawing mode
        if up[0] and not(up[1]):
            if xp == 0 and yp == 0:
                xp = x1
                yp = y1
            print('drawing')
            cv.line(can, (x1, y1), (xp, yp), col, 15)
            xp,yp = x1,y1


    imgGray = cv.cvtColor(can,cv.COLOR_RGBA2GRAY)
    tresh,binary = cv.threshold(imgGray,127,255,cv.THRESH_BINARY_INV)
    binary = cv.cvtColor(binary,cv.COLOR_GRAY2RGB)
    rgb_frame = cv.bitwise_and(rgb_frame,binary)
    rgb_frame =cv.bitwise_or(rgb_frame,can)



    #fin = cv.addWeighted(rgb_frame,0.5,can,0.5,0)
    cv.imshow('img',rgb_frame)
    if cv.waitKey(25) & 0XFF == ord('q'):
        break
  #  print(loc)
