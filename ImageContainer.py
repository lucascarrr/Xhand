import imageio
from Image import Image

class ImageContainer:
    images = []

    def addImage(self, ImageObject):
        self.images.append(ImageObject)
        print(ImageObject.filename, " added!")
    
    def writeImages(self, output_path):
        for image in self.images:
            imageio.imwrite((output_path + image.filename), image.output_image)