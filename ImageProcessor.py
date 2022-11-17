
import cv2 as cv
import numpy as np

#this class contains the methods for processing an image object

class ImageProcessor:
    
    def applyProcessing(self, ImageObject):
        output_image = ImageObject
        output_image = ImageProcessor.histogramNormalization(output_image)
        output_image = ImageProcessor.thresholding(output_image)
        output_image = ImageProcessor.ccAnalysis(output_image)
        output_image = ImageProcessor.findContours(self, output_image)

        return output_image

    def histogramNormalization(ImageObject):
        image = cv.cvtColor(ImageObject, cv.COLOR_BGR2GRAY) #cvt to grey
        image = cv.medianBlur(image, 5)
        return image
   
    def removeSegment(ImageObject):
        width = ImageObject.shape[1]
        height = ImageObject.shape[0]
        
        start_y = height - 200
        for x in range (start_y, height):
            

        

    def thresholding(ImageObject):
        kernel = np.ones((5,5),np.uint8)
        dilation = cv.dilate(ImageObject,kernel,iterations = 5)
        opening = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel)
        image = cv.adaptiveThreshold(opening, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize=7, C=2)
        kernel = np.ones((3,3),np.uint8)
        dilation = cv.dilate(image,kernel,iterations = 2)
        kernel = np.ones((13,13),np.uint8)
        closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
        kernel = np.ones((2,2),np.uint8)
        dilation = cv.dilate(ImageObject,kernel,iterations = 10)
        
        return closing


    def ccAnalysis(ImageObject):
        
        max         = []
        id          = 0
        analysis    = cv.connectedComponentsWithStats(ImageObject, 4, cv.CV_32S)
        output      = np.zeros(ImageObject.shape, dtype="uint8")

        (totalLabels, label_ids, values, centroid) = analysis

        for i in range(0, totalLabels):
            max.append(values[i, cv.CC_STAT_AREA])
        max.sort()

        for i in range(0, totalLabels):
            if values[i, cv.CC_STAT_AREA] == max[totalLabels - 2]:
                id = i
        
        componentMask      = (label_ids == id).astype("uint8") * 255
        cc_analized_image  = cv.bitwise_or(output, componentMask)  
        return cc_analized_image

    def findContours(self, threshed_image):

        # Stores the concaves between fingers
        concave_points  = []
        start_points    = []
        end_points      = []
        hand_points     = []

        contours, hierarchy  = cv.findContours(threshed_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours             = max(contours, key=lambda x: cv.contourArea(x))
        contours2            = map(np.squeeze, contours)

        img = np.zeros([threshed_image.shape[0], threshed_image.shape[1], 3], dtype=np.uint8)
        img[:] = 0

        # Draws the hand contour
        cv.drawContours(img, [contours], -1, (255, 0, 0), 2)
        hull                 = cv.convexHull(contours)
        cv.drawContours(img, [hull], -1, (0, 255, 255), 2)

        return img

