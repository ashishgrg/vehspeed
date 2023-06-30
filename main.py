import cv2
import numpy as np
import pandas as pd


VIDEO_PATH = 'C:/Users/ashis/Desktop/New folder/input/videoplayback.mp4'
OUTPUT_CSV = 'C:/Users/ashis/Desktop/New folder/output/speed.csv'
PIXEL_TO_METERS = 0.1
FPS = 30


vehicle_count = 0
results = []


cap = cv2.VideoCapture(VIDEO_PATH)


bg = cv2.createBackgroundSubtractorMOG2()


while True:

    ret, frame = cap.read()
    if not ret:
        break


    fgmask = bg.apply(frame)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)


            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            cx, cy = x + w // 2, y + h // 2


            if cy > 300 and cy < 400:
                vehicle_count += 1
                results.append([vehicle_count, cx, cy])


    cv2.imshow('Vehicle Speed Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


speeds = []
for i in range(1, len(results)):
    x1, y1 = results[i - 1][1], results[i - 1][2]
    x2, y2 = results[i][1], results[i][2]
    distance_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance_meters = distance_pixels * PIXEL_TO_METERS
    speed = distance_meters * FPS
    speeds.append(speed)


df = pd.DataFrame({'Vehicle': range(1, len(speeds) + 1), 'Speed (m/s)': speeds})
df.to_csv(OUTPUT_CSV, index=False)