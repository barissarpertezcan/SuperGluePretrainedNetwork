import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
img = 255 * np.ones((300, 300, 3))
img = cv2.rectangle(img, (10, 10), (305, 305), (255, 0, 255), 2) # out of bonds hatası vermiyor
cv2.imshow("white", img)
cv2.waitKey(0)
"""

"""
img = cv2.imread("source_img/bottle.jpeg")
print(img.shape)
img = cv2.resize(img, (300, 500), cv2.INTER_AREA)
print(img.shape)
cv2.imshow("bottle", img)
cv2.imwrite("source_img/bottle.png", img)
cv2.waitKey(0)
"""

"""
img = cv2.imread("source_img/bottle2.jpg")
print(img.shape)
img = cv2.resize(img, (300, 500), cv2.INTER_AREA)
print(img.shape)
cv2.imshow("bottle", img)
cv2.imwrite("source_img/bottle2.png", img)
cv2.waitKey(0)
"""

"""
img = cv2.imread("source_img/bottle1.png")
print(img.shape)
img = cv2.resize(img, (640, 480))
print(img.shape)

plt.figure()
plt.imshow(img) 
plt.show()  # display it
"""

# erikli source: (104, 21), (560, 25), (132, 416), (545, 432) : top_left, top_right, bottom_left, bottom_right
# book source: (142, 56), (525, 75), (52, 420), (543, 439) : top_left, top_right, bottom_left, bottom_right

"""
img = cv2.imread("deneme.png")
print(img.shape)
cv2.imshow("deneme", img)
cv2.waitKey(0)

# img = cv2.resize(img, (300, 500), cv2.INTER_AREA)

# if len(img.shape) != 2:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imwrite("deneme.png", img)
# cv2.waitKey(0)
"""

"""
img = cv2.imread("source_img/raw_imgs/mybook.jpeg")
img = cv2.resize(img, (640, 480), cv2.INTER_AREA)
print(img.shape)
cv2.imshow("mybook", img)
cv2.imwrite("source_img/mybook.png", img)
cv2.waitKey(0)
"""

# img = cv2.resize(img, (300, 500), cv2.INTER_AREA)

img = cv2.imread("source_img/mybook.png", 0)
print(img.shape)
cv2.imshow("img", img)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.circle(img, (50, 50), 5, (255, 0, 0), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_back", img)
print(img.shape)

cv2.waitKey(0)