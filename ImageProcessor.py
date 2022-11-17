
import cv2 as cv
import numpy as np

#this class contains the methods for processing an image object
class ImageProcessor:
    
    def applyProcessing(self, ImageObject):
        output_image = ImageObject
        output_image = ImageProcessor.histogramNormalization(output_image)
        output_image = ImageProcessor.thresholding(output_image)
        output_image = ImageProcessor.cleanImage(output_image)
        output_image = ImageProcessor.ccAnalysis(output_image)
        output_image = ImageProcessor.findContours(output_image)
        return output_image

    def histogramNormalization(ImageObject):
        image = cv.cvtColor(ImageObject, cv.COLOR_BGR2GRAY) #cvt to grey
        image = cv.medianBlur(image, 5)
        return image
   
    def getWindowAverage(i, j, ImageObject):
        image_width = ImageObject.shape[1]
        sum = 0

        if j < (image_width // 2):
            for a in range(2):
                for b in range(3):
                    sum += ImageObject[i - b][j + a]
        else:
            for a in range(2):
                for b in range(3):
                    sum += ImageObject[i - b][j - a]
        if sum / 6 > 130:
            return 255
        return 0

    #This method (and getWindowAverage) is required to close-off the hand from the edges of the image (sometimes the thresholding can bleed a little bit)
    def cleanImage(ImageObject):
        image_height  = ImageObject.shape[0]
        image_width   = ImageObject.shape[1]

        # Create an image with the pixels averaged using getWindowAverage
        for i in range(image_height - round(image_height * 0.1), image_height):
            for j in range(image_width):
                ImageObject[i][j] = ImageProcessor.getWindowAverage(i, j, ImageObject)

        return ImageObject       

        
    #The excessive amount of alterations seem to help deal with cases of low contrast, while not breaking images with normal contrast levels
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

    #Connected Component Analysis
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

    def findContours(threshed_image):
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

