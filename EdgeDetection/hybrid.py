import cv2
import numpy as np
import math


def gaussianBlur(img, sigma):
    size = 2 * math.ceil(3 * sigma) + 1
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    gaussian = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    res = conv2d(img, gaussian)
    return res


def conv2d(img, kernel):
    image = cv2.imread(img)
    kernel = cv2.flip(kernel, -1)
    (kernelHeight, kernelWidth) = kernel.shape[:2]
    (imageHeight, imageWidth) = image.shape[:2]
    (paddingHeight, paddingWidth) = (kernelHeight // 2, kernelWidth // 2)
    output = np.zeros(image.shape)
    for y in range(paddingHeight, imageHeight - paddingHeight):
        for x in range(paddingWidth, imageWidth - paddingWidth):
            # If coloured, loop for colours.
            for colour in range(image.shape[2]):
                # Get center pixel.
                center = image[
                         y - paddingHeight: y + paddingHeight + 1, x - paddingWidth: x + paddingWidth + 1, colour
                         ]
                output[y, x, colour] = (center * kernel).sum() / 255
    return output


def lowPassFilter(img, sigma):
    return gaussianBlur(img, sigma)


def highPassFilter(img, sigma):
    return (cv2.imread(img) / 255) - lowPassFilter(img, sigma)


def hybird(imgs, cutoffs):
    low = lowPassFilter(imgs[0], cutoffs[0])
    cv2.imshow('low', low)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('low.jpg', low * 255)
    high = highPassFilter(imgs[1], cutoffs[1])
    cv2.imshow('high', high)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('high.jpg', (high + 0.5) * 255)
    final = low + high
    cv2.imwrite('hybrid.jpg', final)
    cv2.imshow('hybrid', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final


inputImgs = ['littledog.png', 'cat2.png']
inputCutoffs = [5, 5]

hybird(inputImgs, inputCutoffs)
