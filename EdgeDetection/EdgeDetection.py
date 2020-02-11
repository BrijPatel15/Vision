import numpy as np
import math
import cv2
import matplotlib.pyplot as plt



def gaussianBlur(sigma, img):
    size = 2 * math.ceil(3 * sigma) + 1
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    gaussian = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    res = conv2d(img, gaussian)
    return res


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def createSobelFilters(img):
    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = conv2d(img, x)
    Iy = conv2d(img, y)
    res = np.hypot(Ix, Iy)
    res = res / res.max() * 255
    angle = np.arctan2(Iy, Ix)
    return res, angle


def nonMaxSuppression(img, theta):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                neighbour1 = 255
                neighbour2 = 255
                # angle 0
                if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                    neighbour1 = img[i, j + 1]
                    neighbour2 = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    neighbour1 = img[i + 1, j - 1]
                    neighbour2 = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    neighbour1 = img[i + 1, j]
                    neighbour2 = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    neighbour1 = img[i - 1, j - 1]
                    neighbour2 = img[i + 1, j + 1]

                if img[i, j] >= neighbour1 and img[i, j] >= neighbour2:
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except:
                pass
    return Z

# Cannot use cv2.imshow to display images for some reason so I have used matplotlib and it works on Intellij on my
# computer. I have also submitted a python note book where all of this works using cv2.imshow using google colab
def myEdgeDetector(img0, sigma):
    img = cv2.imread(img0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgSmoothed = gaussianBlur(sigma, gray)
    # cv2.imshow('Smooth Img', imgSmoothed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(imgSmoothed)
    plt.show()
    sobelMat, theta = createSobelFilters(imgSmoothed)
    # cv2.imshow('Smooth Img', sobelMat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(sobelMat)
    plt.show()
    nonMaxImg = nonMaxSuppression(sobelMat, theta)
    # cv2.imshow('Non-max', nonMaxImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(nonMaxImg)
    plt.show()
    return nonMaxImg


myEdgeDetector("img0.jpg", 3)
