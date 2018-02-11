'''  Weight Measurement of Sea cucumber (Holothuia Scabra)
        Utilizing image-based volume calculation in an underwater condition

        (c) Polytechnic University of the Philippines
        author - Ryan Pontillas Iraola

        #####################################################################################

        This code primarily for image segmentation using watershed algorithm supported
                with image processing: OTSU THreshold, Normalization, Morphological dilate/erode,
                Euclidian distance transform etc . .
                written in Python language with OPENCV

        ##################################################################################

            --- segment.py ---
'''
import numpy as np               
import cv2
from matplotlib import pyplot as plt

#Reading the image
img = cv2.imread('f.png')

#img = cv2.medianBlur(img, 17)<----- initial blur for denoise; just in case

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Conversion to gray scale
gray = cv2.normalize(gray,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX) #overwrite gray and apply pixel normalization 
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #Threshold applying OTSU

img3 = cv2.medianBlur(img, 21)#jump blurr only for output

cv2.imshow("Otsu", thresh)#output thresh 

# noise removal
kernel = np.ones((3,3),np.uint8)

opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
#opening2 = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 4)

# sure background area
#blurr
#sure_bg = cv2.dilate(opening,kernel,iterations=0)
sure_bg = cv2.erode(opening, kernel, iterations=1)




# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,0)
dist_1 = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,0)
ret, sure_fg = cv2.threshold(dist_1,0.999*dist_1.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)
#unknown = cv2.GaussianBlur(unknown,(9,9),0)

cv2.imshow("Sure bg", sure_bg)
cv2.imshow("Sure fg",sure_fg)
cv2.imshow("unknown",unknown)
contours, hierarchy = cv2.findContours(sure_fg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Creating a numpy array for markers and converting the image to 32 bit using dtype paramter
marker = np.zeros((thresh.shape[0], thresh.shape[1]),dtype = np.int32)

marker = np.int32(sure_fg) + np.int32(sure_bg)


#Marker Labelling
for id in range(len(contours)):
	cv2.drawContours(marker,contours,id,id+2, -1)

marker = marker + 1
marker[unknown==255] = 0
cv2.watershed(img3, marker)
img[marker==1]=(255,255,255)

#shooowwwwwww
cv2.imshow('watershed', img)
cv2.imwrite('img.jpg',img)
img7 = cv2.imread('img.jpg')
img8 = cv2.medianBlur(img7, 9)#blurr
cv2.imshow('img4',img8)
edge = cv2.Canny(gray, 5000, 5000, apertureSize=5)
cv2.imshow('dge',edge)
imgplt = plt.imshow(dist_1)


plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
