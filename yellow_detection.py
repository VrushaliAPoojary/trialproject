import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

while True:
   
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

  
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

  
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

  
    yellow_objects = cv2.bitwise_and(frame, frame, mask=mask)

  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 
    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            
           
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  
    cv2.imshow("Original Frame with Yellow Objects", frame)

    
    cv2.imshow("Yellow Objects", yellow_objects)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
