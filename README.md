# Computer-vision-on-pdf

It explains about computer vision models used for detection and extraction of data especially images and tables from PDF documents and store as png images

This project compared the extraction of tables, images from PyPDF2 and Amazon Textract for processing and using the data to multiple AI applications.

## AWS Services:

 Amazon Web Services like Amazon Textract and AWS Reckognition is used to detect table and extract table in which the data obtained of dimensions in json format.
 The data is further enhanced, processed to extract the image dimensions using Amazon Reckognition that runs based on OCR Tesseract library. The data for pdf file is accessed from S3 bucket and the input data used is pdf file.

## YOLO Models:

 Further, used Yolo model to extract the tables from the pdf document. PyPDF2 library is used to detect the tables and tesseract library is used to extract from the document.

 ## Libraries Used:
 
PyPDF2, OCR Tesseract
