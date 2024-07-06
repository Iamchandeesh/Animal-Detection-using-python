import smtplib
from email.mime.multipart import MIMEMultipart

import cv2
import time
import os
import pyttsx3
from playsound import playsound

classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    print(className)

                    if className=='elephant':

                        time.sleep(0.5)


                        print("elephant")
                        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                        email_addr = 'sarun140689@gmail.com'
                        email_passwd = 'almqpdhzxzerjmqj'
                        server.login(email_addr, email_passwd)
                        server.sendmail(from_addr=email_addr, to_addrs='chandeeshrajagopal@gmail.com',
                                        msg="Animal Detection Alert!!!")
                        server.close()
                        playsound('D:/AnimalDetection/alarm.mp3')

                    elif className=='horse':

                        time.sleep(0.5)
                        #print("horse")
                        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                        email_addr = 'sarun140689@gmail.com'
                        email_passwd = 'almqpdhzxzerjmqj'
                        server.login(email_addr, email_passwd)
                        server.sendmail(from_addr=email_addr, to_addrs='logeshwarans159@gmail.com',
                                        msg="Animal Detection Alert!!!")
                        server.close()
                        #playsound('D:/AnimalDetection/alarm.mp3')
                        playsound('D:/AnimalDetection/alarm.mp3')

                    elif className=='bear':
                        time.sleep(0.5)
                        #print
                        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                        email_addr = 'sarun140689@gmail.com'
                        email_passwd = 'almqpdhzxzerjmqj'
                        server.login(email_addr, email_passwd)
                        server.sendmail(from_addr=email_addr, to_addrs='logeshwarans159@gmail.com',
                                        msg="Animal Detection Alert!!!")
                        playsound('D:/AnimalDetection/alarm.mp3')



    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2,objects=['elephant','horse','bear'])
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)

