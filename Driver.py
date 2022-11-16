import os

import cv2 as cv
import numpy as np

import ImageContainer

#loading images into objects -> image_container
def helper(path, container):
    for filename in os.listdir(path):
        if (filename.endswith(".png")):
            source_img  = np.array(cv.imread(os.path.join(path, filename), 0))
            container.addImage(source_img)

def main():
    input_path  = ""
    output_path = ""
    holder = ImageContainer()
    helper(input_path, holder)


if __name__ == "__main__":
    main()