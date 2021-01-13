# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 00:40:06 2021

@author: yusuf
"""

import cv2 
import numpy as np

img = cv2.imread("image.png")
    

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edged =cv2.Canny(gray,50,200)

contours ,hierarchy = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print("Kontür Sayısı",len(contours))

cv2.drawContours(img ,contours,-1,(255,0,0),4)#-1 tüm konturları çizdirir,255 kontur renk , 4 kontur kalınlık
            
        
cv2.imshow("frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
    