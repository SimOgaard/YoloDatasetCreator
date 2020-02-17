import cv2
import imutils
import os
import random
import numpy as np
import xml.etree.cElementTree as ET

### Darken/lighen photo -> Fixa backgroundremover -> Film to images cropped and background removed -> ?Koden till google colab?

def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.JPEG".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root
 
def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

DESTINATION_DIR = "GeneratingDataset/GeneratedImagesXml"

backgroundShape = (224,224)
minSize = 32
maxSize = 112
imageStart = 1000
itterations = 1500
maxCharactersAllowed = 5
total = imageStart

for i in range(itterations):

    charactersOnCanvas = []
    backgroundDir = "Get Images/BackgroundImages/"+random.choice(os.listdir("C:/Users/s8simoga/Documents/GitHub/Livindead3/YoloDatasetCreator/YoloDataset/lego-gubbar-detection/Get Images/BackgroundImages"))
    imgBackground = cv2.cvtColor(cv2.imread(backgroundDir), cv2.COLOR_RGB2RGBA)
    resizedBackground = cv2.resize(imgBackground, backgroundShape)

    charactersAmount = random.randint(1,maxCharactersAllowed)
    for amounts in range(charactersAmount):
        skip = False
        characterDir = "Edit Characters/FramesNoBackground/"+random.choice(os.listdir("C:/Users/s8simoga/Documents/GitHub/Livindead3/YoloDatasetCreator/YoloDataset/lego-gubbar-detection/Edit Characters/FramesNoBackground/"))
        imgCharacter = cv2.imread(characterDir, cv2.IMREAD_UNCHANGED)

        rotation = random.randint(0,360)
        rotatedCharacter = imutils.rotate_bound(imgCharacter, rotation)

        largeSide = max([rotatedCharacter.shape[1], rotatedCharacter.shape[0]])
        maxScale = int((maxSize/largeSide)*100)
        minScale = int((minSize/largeSide)*100)
        scalePercent = random.randint(minScale, maxScale)
        resizedCharacter = cv2.resize(rotatedCharacter, (int(rotatedCharacter.shape[1]*scalePercent/100),int(rotatedCharacter.shape[0]*scalePercent/100)))

        y_offset = random.randint(0,backgroundShape[1]-resizedCharacter.shape[0])
        x_offset = random.randint(0,backgroundShape[0]-resizedCharacter.shape[1])
        y1, y2 = y_offset, y_offset + resizedCharacter.shape[0]
        x1, x2 = x_offset, x_offset + resizedCharacter.shape[1]

        alpha_s = resizedCharacter[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        if charactersOnCanvas:
            for _, x1_, y1_, x2_, y2_ in charactersOnCanvas:
                if x2_ < x1 or x2 < x1_ or y2_ < y1 or y2 < y1_:
                    continue                
                skip=True
                break
            
        if skip:
            continue

        for c in range(0, 3):
            resizedBackground[y1:y2, x1:x2, c] = (alpha_s * resizedCharacter[:, :, c] + alpha_l * resizedBackground[y1:y2, x1:x2, c])

        charactersOnCanvas.append(["lego gubbe",x1, y1, x2, y2])

    file_prefix = str(total).zfill(8)
    root = create_root(file_prefix, backgroundShape[0], backgroundShape[1])
    root = create_object_annotation(root, charactersOnCanvas)
    tree = ET.ElementTree(root) 
    tree.write("{}/{}.xml".format(DESTINATION_DIR, file_prefix))

    gauss = np.random.normal(0,random.randint(0,250)/1000,resizedBackground.size)
    gauss = gauss.reshape(resizedBackground.shape[0],resizedBackground.shape[1],resizedBackground.shape[2]).astype('uint8')
    noise = (resizedBackground + resizedBackground * gauss) # * (random.randint(10,50)/10000)

    # cv2.imshow('noise', noise)
    # k = cv2.waitKey()
    # if k==27:
    #     break

    # cv2.imwrite("GeneratingDataset/GeneratedImages/"+"{}.JPEG".format(str(total).zfill(8)), resizedBackground)
    cv2.imwrite("GeneratingDataset/GeneratedImages/"+"{}.JPEG".format(file_prefix), noise)
    total+=1

cv2.destroyAllWindows()
