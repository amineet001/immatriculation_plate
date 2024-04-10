# Introduction

This project is aimed at training a deep learning model to detect license plates in images. It involves creating a dataset of license plates images and their corresponding coordinates, training two different models (***Faster-RCNN*** and ***YOLOv5***), and an Android application that can be used to test the trained models.
To generate the dataset of license plate images and their corresponding coordinates, you should run the following commands in your terminal

    $ scrapy runspider Data_Generator/scrapy.py
    $ python3 Generator.py

# Requirements
Here are the libraries and their versions needed for this project:

- torch (version 1.8.1 or later)
- torchvision (version 0.9.1 or later)
- easyocr (version 1.3.2 or later)
- opencv-python (version 4.5.2.54 or later)
- matplotlib (version 3.4.2 or later)
- numpy (version 1.20.3 or later)
- pandas (version 1.2.4 or later)
- tqdm (version 4.61.1 or later)

# Data Generator

The data_generator folder contains two files, ***scrapy.py*** and ***Generator.py***. The ***scrapy.py*** file is used to scrape license plate images from the internet. The ***Generator.py*** file takes the scraped images and creates a training database in the image folder with corresponding .txt files containing the coordinates of the license plates in COCO format.


# Models

Two different deep learning models were trained: ***Faster-RCNN*** and ***YOLOv5*** . The ***Faster-RCNN*** model can be found in the FRCNN folder, and the ***YOLOv5*** model can be found in the yolov5 folder.

# Android App

The AndroidApp folder contains an Android application that can be used to test the trained models. To use the application, add test images (***test1.png***, ***test2.png***, ***test3.png***) to the folder and run the application. The application will load the trained models and use them to detect license plates in the test images.

# Conclusion

This project provides a good starting point for developing a license plate detection system using deep learning. It includes data scraping, dataset creation, model training, and an Android application for testing the trained models.

