import cv2
import numpy as np
from time import sleep

l_min=80 #min length of rectangle
h_min=80 #min height of rectangle

offset=6 #Permissible error between pixwel 

pos_line=550 #position of line

delay= 60 #FPS 

detect = []
cars = 0

	
def centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('4kvideo.mp4')
subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret ,frame = cap.read()
    temp = float(1/delay)
    sleep(temp) 
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract.apply(blur)
    dilation = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilation, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    outline,h = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, pos_line), (1200, pos_line), (255,127,0), 3) 
    for(i,c) in enumerate(outline):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_outline = (w >= l_min) and (h >= h_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        
        centre1 = centre(x, y, w, h)
        detect.append(centre1)
        cv2.circle(frame, centre1, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(pos_line+offset) and y>(pos_line-offset):
                cars+=1
                cv2.line(frame, (25, pos_line), (1200, pos_line), (0,127,255), 3)  
                detect.remove((x,y))
                print("car is detected : "+str(cars))        
       
    cv2.putText(frame, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame)
    cv2.imshow("Detected",dilated)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
