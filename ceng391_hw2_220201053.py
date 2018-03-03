import cv2
import numpy

def gaussian(x,sigma):
    return (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = numpy.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(numpy.round(filtered_image))
    return new_image

image = cv2.imread("in_img.jpg",0)
filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
image_own = bilateral_filter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_own.png", image_own)

"""
The bilateral filter is controlled by important parameters. Two of them are sigma values.
Generally, the bilateral filter gives us more control over image.
If we increment both sigma values at the same time, the bigger sigma values gives us a
more blurred image. If we give sigma values near zero, smoothing does not occur. Changing sigma i
directly affects the blur effect on the image. However, sigma s does not affect blur rate. There
is no big effect on the image after changing only the sigma s. Sharpness does not necessary that
much with sigma s rather than sigma i. To have a more blurred image, we should take sigma values
bigger.
"""