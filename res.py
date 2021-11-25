# This code was used to resize and then crop the pictures of hands taken with my phone.
# Once they have the right size (300*300 pixels), they were uploaded to Github
import cv2

for i in range (1, 10):
    url_load = 'E:\ALEJANDRO\DATA_SCIENCE\TENSORFLOW\MANOS\Originales\M' + str(i) + '.jpg'
    print(url_load)
    img = cv2.imread(url_load)
    print(img.shape)

    # 1200*1600 pixels
    w_percent = 4
    h_percent = 4
    width = int(img.shape[1]/w_percent)
    height = int(img.shape[0]/h_percent)
    dimension = (width, height)
    resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)

    print(resized.shape)
    cv2.imshow('Salida', resized)

    c_img = resized[100:400, 0:300] # Slicing to crop the image
    print(c_img.shape)
    cv2.imshow('Salida Crop', c_img)

    # cv2.waitKey(0)

    url_save = 'E:\ALEJANDRO\DATA_SCIENCE\TENSORFLOW\MANOS\Crop\M' + str(i) + '.jpg'
    cv2.imwrite(url_save, c_img)

    print(url_save)
    
# Alejandro García Lagos
