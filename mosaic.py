import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore k-means convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
        image = image.tolist()
        averaged_image = []
        for y in range(0, len(image)-1, 2):
            averaged_image.append([])
            for x in range(0, len(image[0])-1, 2):
                r = 0
                b = 0
                g = 0
                r, b, g = self.add_rbg_4(r,b,g,image,y,x)
                r, b, g = [round(r/4), round(b/4), round(g/4)]
                averaged_image[int(y/2)].append([int(r),int(b),int(g)])
                
        averaged_image = np.array(averaged_image,dtype=np.uint8)
        return averaged_image
    def get_dominant_color_image1(self, image):
        image = image.tolist()
        dominant_image = []
        for y in range(0, len(image)-1, 2):
            dominant_image.append([])
            for x in range(0, len(image[0])-1, 2):
                # Collect the 4 pixels
                pixels = [
                    image[y][x],
                    image[y][x+1],
                    image[y+1][x],
                    image[y+1][x+1]
                ]
                # Find the dominant color (most frequent)
                pixels = [tuple(pix) for pix in pixels] # make them hashable
                counts = {}
                for pix in pixels:
                    counts[pix] = counts.get(pix, 0) + 1
                dominant = max(counts.items(), key=lambda x: x[1])[0]
                dominant_image[int(y/2)].append(list(dominant))
                    
        dominant_image = np.array(dominant_image, dtype=np.uint8)
        return dominant_image
    def get_dominant_color_image_with_kmeans(self, image, k=1):
        image = image.tolist()
        dominant_image = []
        
        for y in range(0, len(image)-1, 2):
            dominant_image.append([])
            print(f"\rProgress: {y/len(image)*100}%", end='', flush=True)
            for x in range(0, len(image[0])-1, 2):                
                # Collect the 4 pixels
                pixels = [
                    image[y][x],
                    image[y][x+1],
                    image[y+1][x],
                    image[y+1][x+1]
                ]
                
                # Convert to NumPy array for k-means compatibility
                pixels = np.array(pixels)
                
                # Perform k-means clustering to find the dominant color
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(pixels)  # Fit to the 4 pixels
                dominant_color = kmeans.cluster_centers_[0]  # The center of the first cluster
                
                # Append the dominant color (rounded to integers)
                dominant_image[int(y/2)].append(list(map(int, dominant_color)))    
        dominant_image = np.array(dominant_image, dtype=np.uint8)
        return dominant_image
    def get_dominant_color_image_with_kmeans_blocks(self, image, block_size=4, k=1):
        """
        block_size: Size of block, e.g., 4 for 4x4
        k: Number of clusters inside each block (usually 1 or 2)
        """
        height, width, _ = image.shape

        # Make sure the dimensions are divisible by block_size
        new_width = width - (width % block_size)
        new_height = height - (height % block_size)
        image = cv.resize(image, (new_width, new_height))
        
        result_image = []
        
        for y in range(0, new_height, block_size):
            print(f"\rProgress: {round(y/len(image)*100,2)}%", end='', flush=True)
            result_image.append([])
            for x in range(0, new_width, block_size):
                # Collect all pixels in the block
                block = image[y:y+block_size, x:x+block_size].reshape(-1, 3)
                
                # Perform k-means clustering on the block
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=5)
                kmeans.fit(block)
                
                # Take the dominant color (center of biggest cluster)
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                dominant_cluster_idx = labels[np.argmax(counts)]
                dominant_color = kmeans.cluster_centers_[dominant_cluster_idx]
                
                result_image[int(y/block_size)].append(list(map(int, dominant_color)))
        
        result_image = np.array(result_image, dtype=np.uint8)
        return result_image
    def create_average_image(self,image,loops):
        average_image = image
        for i in range(loops):
            average_image = self.get_averaged_image1(average_image)
        return average_image
    def create_dominant_color_image(self,image,loops):
        average_image = image
        for i in range(loops):
            average_image = self.get_dominant_color_image1(average_image)
        return average_image     
    def create_dominant_kmeans_image(self,image,loops):
        average_image = image
        for i in range(loops):
            average_image = self.get_dominant_color_image_with_kmeans(average_image)
        return average_image 
    def find_nearest_color(self,color, palette):
        color = np.array(color)
        palette = np.array(palette)
        distances = np.linalg.norm(palette - color, axis=1)
        nearest_idx = np.argmin(distances)
        return palette[nearest_idx]
    def get_pre_color_image(self,dominant_image,palette):
        # Snap every pixel in downscaled image
        snapped_image = []
        for row in dominant_image:
            snapped_row = []
            for color in row:
                snapped_color = self.find_nearest_color(color, palette)
                snapped_row.append(list(snapped_color))
            snapped_image.append(snapped_row)

        snapped_image = np.array(snapped_image, dtype=np.uint8)
        return snapped_image

#Some example running               
# if __name__ == "__main__":
#     m = Mosaic()
#     img = m.load_image("test_img_dog.jpg")
#     resized_image = m.resize_2(img)
#     # averaged_image = m.create_average_image(resized_image,4)
#     # averaged_image = m.create_dominant_color_image(resized_image,4)
#     # averaged_image = m.create_dominant_kmeans_image(resized_image,1)
#     averaged_image = m.get_dominant_color_image_with_kmeans_blocks(img,block_size=12,k=2)
#     tile_palette = [
#         [0, 0, 0],        # black
#         [255, 0, 0],      # red
#         [0, 255, 0],      # green
#         [0, 0, 255],      # blue
#         [255, 255, 0],    # yellow
#     ]
#     averaged_image = m.get_pre_color_image(averaged_image,tile_palette)
#     # Plotting
#     fig, ax = plt.subplots(1, 2, figsize=(16, 8))
#     fig.tight_layout()

#     # Display manipulated 
#     ax[0].imshow(cv.cvtColor(averaged_image, cv.COLOR_BGR2RGB))
#     ax[0].set_title("Manipulated image")

#     #Display original
#     ax[1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#     ax[1].set_title("Original image")

#     plt.show()
