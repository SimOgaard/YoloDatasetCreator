import cv2
img = cv2.imread("GeneratingDataset/GeneratedImages/00000000.JPEG")
cv2.rectangle(img, (296, 446), (425, 507), (255,0,0), 1)
cv2.imshow("img", img)
# [144.0, 66.0, 243.0, 267.0]
cv2.waitKey()