#!/usr/bin/env python
# coding: utf-8

# # Auto Word Collect Phase 1
# 
# By Nur Hazim Mohamad Nor
# 
# Goal: To suggest possible words to solve the Word Collect game
# 
# Word Collect is a mobile game where the goal is to form words based on the letters given.
# 
# This project will be split to a few parts:
# Phase 1: Word suggestion is done on the PC
# Phase 2: Word suggestion is done on an Android device
# Phase 3: Live suggestion using a smartphone camera??

# For Phase 1, we'll further break it down to a few parts:
# 
# Part 1: Detect tiles from the screenshots provided & crop tiles <br>
# Part 2: Feed into an OCR<br>
# Part 3: Get an English library and suggest words from there

# In[1]:


from detecto import core, utils, visualize
import glob, os
import torch
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\msys64\mingw64\bin\tesseract.exe'


# ## Part 1: Detect tiles from the screenshots provided & crop tiles

# Reference: https://towardsdatascience.com/build-a-custom-trained-object-detection-model-with-5-lines-of-code-713ba7f6c0fb
# 

# In[4]:


# Getting all the train images
path = 'images/'
images = []
for file in os.listdir(path):
    if file.endswith(".png"):
        images.append(file)


# In[6]:


images


# In[7]:


# Testing out the pre-trained model to see if it detects anything in the screenshot
image = utils.read_image(path + images[0])
model = core.Model()

labels, boxes, scores = model.predict_top(image)
visualize.show_labeled_image(image, boxes, labels)


# The pre-trained model failed as expected. Though it did detect one of the tiles as a 'book'.

# << Label screenshots with labelImg >>

# In[10]:


# Train model with custom labels
dataset = core.Dataset(path)
model = core.Model(['tile'])
model.fit(dataset)


# In[5]:


# Getting all the test images
path = 'images/test/'
images = []
for file in os.listdir(path):
    if file.endswith(".png"):
        images.append(file)
images


# In[6]:


tiles = dict()
for image in images:
    curr_image = utils.read_image(path + image)
    predictions = model.predict(curr_image)

    labels, boxes, scores = predictions
    tiles[image] = boxes
    visualize.show_labeled_image(curr_image, boxes, labels)


# In[7]:


tiles


# In[12]:


# [ 575.8508, 1701.7289,  860.3646, 2023.0233]
def conv_tensor_to_opencv(tensor):
    temp_list = list(map(int, tensor.tolist()))
    i = 30
    return slice(temp_list[1] + i,
                 temp_list[3] - i), \
           slice(temp_list[0] + i,
                 temp_list[2] - i)


# In[15]:


test_img = cv2.imread(path + images[0])
# crop_img = test_img[tiles[images[0]][0].tolist()]
crop_to = conv_tensor_to_opencv(tiles[images[0]][0])
crop_img = test_img[crop_to]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


# In[3]:


# model.save('tile_detector.pth')
# model = core.Model.load('tile_detector.pth', ['tile'])


# ## Part 2: Feed into an OCR

# In[75]:


def preprocess(image):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # draw the correction angle on the image so we can validate it
#     cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     ret, labels = cv2.connectedComponents(rotated)
#     for label in range(1,ret):
#     mask = np.array(labels, dtype=np.uint8)
#     mask[labels == 1] = 255

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1

    min_size = max(sizes)
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255 
            
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(img2 > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


# In[36]:


# cv2.imshow("cropped", preproceess(crop_img))
# cv2.waitKey(0)


# In[35]:


# ret, labels = cv2.connectedComponents(preproceess(crop_img))
# for label in range(1,ret):
# mask = np.array(labels, dtype=np.uint8)
# mask[labels == 1] = 255
# cv2.imshow('component',mask)
# cv2.waitKey(0)


# In[50]:


# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(preproceess(crop_img), connectivity=8)
# sizes = stats[1:, -1]; 
# nb_components = nb_components - 1

# min_size = max(sizes)
# img2 = np.zeros((output.shape))
# for i in range(0, nb_components):
#     if sizes[i] >= min_size:
#         img2[output == i + 1] = 255 
        
# cv2.imshow('component',mask)
# cv2.waitKey(0)


# In[76]:


tess_config = ' --psm 10 --tessdata-dir "C:/msys64/mingw64/share/tessdata/"'

# i = 0
char = dict()
for screenshot, tile_list in tiles.items():
    test_img = cv2.imread(path + screenshot)
    letters = []
    for tile in tile_list:
        crop_to = conv_tensor_to_opencv(tile)
        crop_img = preprocess(test_img[crop_to])
#         crop_img = test_img[crop_to]
        

#         cv2.imwrite( "images/cropped/cropped_" + str(i) + ".jpg", crop_img);
#         i += 1
        letter = pytesseract.image_to_string(crop_img, lang='eng', 
                                             config=tess_config)
#         print(letter)
        cv2.putText(crop_img, letter,
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        letters.append(letter)
    char[screenshot] = letters
char


# Progress check: Tesseract gets about 3 of the characters wrong right with the 4 sample images. 
# 2 of them add in additional characters (TV, Ee:) and 1 of them classified F as E
# 
# Next step: Get data and train a custom model for OCR

# In[ ]:




