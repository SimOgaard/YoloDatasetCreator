import cv2
import numpy as np
import glob
from PIL import Image

BLUR = 11
CANNY_THRESH_1 = 240
CANNY_THRESH_2 = 255
MASK_DILATE_ITER = 9
MASK_ERODE_ITER = 9
MASK_COLOR = (0.0,0.0,0.0)

total = 0

for image in glob.glob("Get Images/Images/*.jpg"):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    # edges = cv2.dilate(edges, None)
    # edges = cv2.erode(edges, None)
    edges = cv2.dilate(edges, None, iterations=MASK_DILATE_ITER)
    edges = cv2.erode(edges, None, iterations=MASK_ERODE_ITER)
    # mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    # mask = cv2.inRange(gray, CANNY_THRESH_1, CANNY_THRESH_2)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was: 
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #-- Smooth mask, then blur it --------------------------------------------------------
    # mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    # mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    mask_stack = np.dstack([mask]*4)
    mask_stack  = mask_stack.astype('float') / 255.0 
    newimg = cv2.multiply(mask_stack, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA).astype("float")/255)
    cords = Image.fromarray(newimg.astype('uint8'), 'RGBA').getbbox()
    newimg = newimg[cords[1]:cords[3],cords[0]:cords[2]]

    cv2.imwrite("Edit Characters/ImagesNoBackground/"+"{}.png".format(str(total).zfill(8)), (newimg * 255).astype('uint8'))
    total+=1
