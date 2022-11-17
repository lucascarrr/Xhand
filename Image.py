import numpy as np
from ImageProcessor import ImageProcessor


class Image:
    filename        = ""
    rotation_factor = 0

    original_image  = np.array(0)
    output_image    = np.array(0)

    finger_lengths  = []
    countours       = []

    def __init__(self, original_image, filename):
        self.original_image = original_image
        self.filename = filename
        self.processImage()
    
    #pipeline should be:
    # normalization -> thresholding ->  
    def processImage(self):
        self.output_image = self.original_image
        self.output_image = ImageProcessor.applyProcessing(self, self.output_image)

   

        
    


