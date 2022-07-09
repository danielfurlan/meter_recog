import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
from numpy import asarray
from keras.preprocessing import image

def to_gray(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2_imshow(gray)
  return gray


# Adaptive Thresholding: This method gives a threshold for a small part of the image
def binarize(img):
  img2 = img.copy()
  
  if len(img2.shape) == 3:
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  imgf1 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #imgf contains Binary image
  print("Binarized")
  cv2_imshow(imgf1)
  # # Otsu’s Binarization: This method gives a threshold for the whole image
  # ret, imgf2 = cv2.threshold(img2, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU) #imgf contains Binary image
  # cv2_imshow(imgf2)
  return imgf1


def correct_skew(image, delta=1, limit=45):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    gray = image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(gray.shape) == 3:
      gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
      
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    # (h, w) = image.shape[:2]
    (h, w) = gray.shape[:2]
    # print("Our original h , w : ", h, w, center)
    # if h > w:
    #   h, w = w, h
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    # if h > w:
    #   print("Rotating the image!!")
    #   corrected = cv2.rotate(corrected.copy(), cv2.cv2.ROTATE_90_CLOCKWISE)
    return best_angle, corrected

def thinning(img):
# erosion is used for pulling out small white noise
  kernel = np.ones((5,5),np.uint8)
  correct = img.copy()
  erosion = cv2.erode(correct,kernel,iterations = 1)
  print("Right  : Thinning")
  plt.subplot(121), plt.imshow(img) 
  plt.subplot(122), plt.imshow(erosion) 
  plt.show()
  # cv2_imshow(erosion)
  return erosion


def noise_removal(img):

# denoising of image saving it into dst image 
  dst = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 100, 10, 10, 7, 15) 
# Plotting of source and destination image 
  print("Right : Noise removed")
  plt.subplot(121), plt.imshow(img) 
  plt.subplot(122), plt.imshow(dst) 
  plt.show()
  # print("teype of noise removal : ", type(dst))
  return image.img_to_array(dst, dtype='uint8')
