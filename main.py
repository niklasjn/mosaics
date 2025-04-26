import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
class Mosaic:
    def __init__(self):
        pass # for now
    def load_image(self,path):
        image = cv.imread(path)
        return image
    def resize_2(self,image):
        new_width = len(image[0]) - (len(image[0]) % 2)
        new_height = len(image) - (len(image) % 2)
        resized_image = cv.resize(image, (new_width, new_height))
        return resized_image
    def add_rbg_pixel(self,r,b,g,pixel):
        r += pixel[0]
        b += pixel[1]
        g += pixel[2]
        return [r,b,g]
    def add_rbg_4(self,r,b,g,image,y,x):
        r,b,g = self.add_rbg_pixel(r,b,g,image[y][x])
        r,b,g = self.add_rbg_pixel(r,b,g,image[y][x+1])
        r,b,g = self.add_rbg_pixel(r,b,g,image[y+1][x])
        r,b,g = self.add_rbg_pixel(r,b,g,image[y+1][x+1])
        return [r,b,g]
    def get_averaged_image1(self,image):
        averaged_image = [[[-1,-1,-1]]*int(len(image[0])/2)]*int(len(image)/2)
        for y in range(0, len(image)-1, 2):
            r = 0
            b = 0
            g = 0
            for x in range(0, len(image[0])-1, 2):
                r, b, g = self.add_rbg_4(r,b,g,image,y,x)
                r, b, g = [round(r/4), round(b/4), round(g/4)]
                averaged_image[int(y/2)][int(x/2)] = [int(r),int(b),int(g)]
        averaged_image = np.array(averaged_image,dtype=np.uint8)
        return averaged_image
                
if __name__ == "__main__":
    m = Mosaic()
    img = m.load_image("test_img_dog.jpg")
    resized_image = m.resize_2(img)
    averaged_image = m.get_averaged_image1(resized_image)
    print("-----image------")
    print(img[0])
    print("------average------")
    print(averaged_image[0])

    print(len(img))
    print(len(averaged_image))

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.tight_layout()

    # Display manipulated 
    ax[0].imshow(cv.cvtColor(averaged_image, cv.COLOR_BGR2RGB))
    ax[0].set_title("Manipulated image")

    #Display original
    ax[1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax[1].set_title("Original image")

    plt.show()
