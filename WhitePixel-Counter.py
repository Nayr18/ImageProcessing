import cv2
import numpy as np

'''
    Python Opencv, WHITE PIXEL COUNTER 
    author: Ryan Pontillas Iraola

'''

img1 = cv2.imread('say1.png')
img2 = cv2.imread('say2.jpg')
img3 = cv2.imread('say3.jpg')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY)
ret, thresh3 = cv2.threshold(gray3,0,255,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
opening1 = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 1)
opening2 = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel, iterations = 1)
opening3 = cv2.morphologyEx(thresh3,cv2.MORPH_OPEN,kernel, iterations = 1)

contours, hierarchy = cv2.findContours( opening1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # draw rectangle around contour on original image
    cv2.rectangle(img3,(x,y),(x+w,y+h),(255,0,255),2)
    print "counter 1 height: ",h

contours, hierarchy = cv2.findContours( opening2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # draw rectangle around contour on original image
    cv2.rectangle(img3,(x,y),(x+w,y+h),(255,0,255),2)
    print "counter 2 height: ",h

contours, hierarchy = cv2.findContours( opening3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # draw rectangle around contour on original image
    cv2.rectangle(img3,(x,y),(x+w,y+h),(255,0,255),2)
    print "counter 3 height: ",h


            
print "counter 1: ",cv2.countNonZero(opening1)
print "counter 2: ",cv2.countNonZero(opening2)
print "counter 3: ",cv2.countNonZero(opening3)
