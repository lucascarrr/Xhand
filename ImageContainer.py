import imageio
from Image import Image

#container for all our image objects
class ImageContainer:
    images = []

    def addImage(self, ImageObject):
        self.images.append(ImageObject)
        print(ImageObject.filename, " added!")
    
    def writeImages(self, output_path):
        for image in self.images:
            imageio.imwrite((output_path + image.filename), image.output_image)