
import math
import cv2 as cv
import numpy as np

#this class contains the methods for processing an image object
class ImageProcessor:
    
    def applyProcessing(self, ImageObject):
        output_image = ImageObject
        final = ImageObject
        output_image = ImageProcessor.histogramNormalization(output_image)
        output_image = ImageProcessor.thresholding(output_image)
        output_image = ImageProcessor.cleanImage(output_image)
        output_image = ImageProcessor.ccAnalysis(output_image)
        output_image, final = ImageProcessor.findContours(output_image, final)
        return output_image

    def histogramNormalization(img):
        img=cv.subtract(img, np.average(img.flatten()))
        clahe = cv.createCLAHE(clipLimit=15)
        img = clahe.apply(img)
        return img
        
    def thresholding(ImageObject):
        kernel = np.ones((3,3),np.uint8)
        image = ImageObject
        for i in range (20):
            image = cv.dilate(ImageObject,kernel,iterations = 1)
            image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

        #blur   = cv.GaussianBlur(image, (5, 5), 0)
        ret, threshold = cv.threshold(image, 25, 255, cv.THRESH_BINARY)       
        blur   = cv.GaussianBlur(threshold, (9, 9), 0)

        return blur


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

    def findContours(ImageObject, final_image):
        # Stores the concaves between fingers
        concave_points  = []
        start_points    = []
        end_points      = []
        hand_points     = []
        
        contours, hierarchy  = cv.findContours(ImageObject, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for i in range (len(contours)):
            cnt = contours[i];
            cv.approxPolyDP(cnt, 10, True);
        

        contours = max(contours, key=lambda x: cv.contourArea(x))
        contours2 = map(np.squeeze, contours)

        img = np.zeros([ImageObject.shape[0], ImageObject.shape[1], 3], dtype=np.uint8)
        img[:] = 0

        # Draws the hand contour
        cv.drawContours(img, [contours], -1, (255, 0, 0), 2)
        hull                 = cv.convexHull(contours)
        cv.drawContours(img, [hull], -1, (0, 255, 255), 2)

        #MASKING
        final_image  = ImageProcessor.maskImage(255, [0] * 3, contours, final_image)

        # Find and draws the convexity defects
        hull = cv.convexHull(contours, returnPoints=False)
        defects = cv.convexityDefects(contours, hull)

         # Calculate the angle between points of interest
        for count, i in enumerate(range(defects.shape[0])):
            s, e, f, d  = defects[i][0]
            start       = tuple(contours[s][0])
            end         = tuple(contours[e][0])
            far         = tuple(contours[f][0])
            a           = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b           = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c           = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle       = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))  

            # If angle is less than 90 degrees, treat as fingers
            if angle <= np.pi / 2:  
                concave_points.append(far)
                start_points.append(start)
                end_points.append(end)

            elif angle > (2.5):
                if np.abs(start[0] - far[0]) > 50:
                    hand_points.append(far)

        # Store calculated hand data points of interest
        property_concaves  = concave_points
        property_fingers   = ImageProcessor.computeFingertips(start_points, end_points)

        # Draws points of interest onto image (primarily needed for testing)
        # for element in concave_points:
        #     cv.circle(img, element, 10, [0, 0, 255], -1)
        # for element in start_points:
        #     cv.circle(img, element, 10, [0, 255, 0], -1)
        # for element in end_points:
        #     cv.circle(img, element, 10, [255, 0, 255], -1)

        #middle fingers
        # cv.circle(img, start_points[0], 10, [255, 0, 255], -1)
        # cv.circle(img, end_points[3], 10, [255, 255, 255], -1)
        middle_finger = ImageProcessor.midPoint(start_points[0], end_points[3])
        cv.circle(img, middle_finger, 10, [0, 0, 255], -1)


        #2 concaves on either side of finger
        # cv.circle(img, concave_points[3], 10, [255, 0, 255], -1)
        # cv.circle(img, concave_points[0], 10, [0, 255, 255], -1)
        mid_concave = ImageProcessor.midPoint(concave_points[3], concave_points[0])
        cv.circle(img, mid_concave, 10, [255, 0, 0], -1)

        contours_midpoint = mid_concave
        middle_finger_X = middle_finger[0]
        middle_finger_Y = middle_finger[1]

                # Finds height and width of image
        (h, w)             = img.shape[:2]
        # Anchors the image at the concave of the middle finger for rotation
        (cX, cY)           = (int(contours_midpoint[0]), int(contours_midpoint[1]))

        opp                = abs(middle_finger_X - contours_midpoint[0])
        middleFinger       = [middle_finger_X, middle_finger_Y]
        adj                = math.dist(middleFinger, contours_midpoint)
        rotationAngle      = math.degrees(math.atan(opp / adj))

        # Determines the direction to rotate the image
        if middle_finger_X < contours_midpoint[0]:
            rotationAngle  = 360 - rotationAngle

        # Performs the actual roation of image
        M        = cv.getRotationMatrix2D((cX, cY), rotationAngle, 1.0)
        rotated  = cv.warpAffine(img, M, (w, h))
        rotated  = cv.warpAffine(final_image, M, (w, h))
        
        
        
        return rotated, final_image

    def maskImage(mask_values, fill_color, contours, image):
        
        image       = cv.cvtColor(image, cv.COLOR_BAYER_BG2BGR)
        stencil     = np.zeros(image.shape[:-1]).astype(np.uint8)

        cv.fillPoly(stencil, np.array([contours], dtype="int32"), mask_values)

        sel         = stencil != mask_values
        # Fills all values not in the mask with fill_color
        image[sel]  = fill_color

        return image

    def computeFingertips(start_pts, end_pts):
        # Stores the co-ords of the finger tips
        fingertip_list = []

        try:
            fingertip_list.append(end_pts[1])
            fingertip_list.append(ImageProcessor.midPoint(end_pts[0], start_pts[1]))
            fingertip_list.append(ImageProcessor.midPoint(end_pts[3], start_pts[0]))
            fingertip_list.append(ImageProcessor.midPoint(end_pts[2], start_pts[3]))
            fingertip_list.append(start_pts[2])
        except:
            "error"

        return fingertip_list
        
    def midPoint(a, b):
        return (np.abs((a[0] + b[0]) // 2), np.abs((a[1] + b[1]) // 2))