import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_saliency(img, point, img_size=224):
    saliency = np.zeros((img_size, img_size))
    for p in point:
        saliency[p[0], p[1]] = 100
    saliency = cv2.GaussianBlur(saliency, (51, 51), 9)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    # saliency = cv2.resize(saliency, (img.shape[1], img.shape[0]))
    return saliency

if __name__ == "__main__":
    img = np.zeros((224, 224, 3))
    point = [(100, 100), (200, 200), (50, 50), (150, 150), (75, 75)]
    saliency = create_saliency(img, point)
    cv2.imwrite('saliency.jpg', saliency)
    # print(saliency.shape)