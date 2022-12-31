import urllib.request
import time
import numpy as np
import cv2
import signal

#Thanks to FREEDOMTECH for the basis of the code
#Additional modifications made by the ECE4300 Group C Team

def timeout(signum, frame):
    raise Exception("Cam Timeout")
   
def cam():
    url='http://192.168.1.220/800x600.jpg'
    signal.signal(signal.SIGALRM, timeout) #create timeout signal
    signal.alarm(10)                       #timeout after 10s
    
    try:
        #Use urllib to get the image from the IP camera
        imgResp = urllib.request.urlopen(url)
        
        #Convert image into an numpy array
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        
        #decode array into cv2 image
        image = cv2.imdecode(imgNp,-1)
        
        signal.alarm(0) #turn off alarm
        return image
        
    except:
        image = np.zeros((800, 600, 3), dtype=np.uint8) #create black image
        return image