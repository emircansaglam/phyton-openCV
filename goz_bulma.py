import cv2
import numpy as np
goz_data = 'goz_veri_seti.xml'  
yuz_data = 'yuz_veri_seti.xml'  
yuz_ = cv2.CascadeClassifier(yuz_data)
goz_= cv2.CascadeClassifier(goz_data)
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
        yuz = yuz_.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            
        )
        
        if len(yuz) > 0:
            for (x, y, w, h) in yuz:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_tmp = img[yuz[0][1]:yuz[0][1] + yuz[0][3], yuz[0][0]:yuz[0][0] + yuz[0][2]:1, :]
            frame = frame[yuz[0][1]:yuz[0][1] + yuz[0][3], yuz[0][0]:yuz[0][0] + yuz[0][2]:1]
            goz = goz_.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                
            )
            if len(goz) == 0:
                print('göz kırptı..')
            else:
                print('gözler açık..')
            frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('yüz tanıma', frame_tmp)
        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            break
