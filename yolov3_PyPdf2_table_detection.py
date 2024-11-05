#%%
import os
import copy
from camelot import io as camelot
import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PyPDF2
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfReader
from pdf2image import convert_from_path, convert_from_bytes
from utils.detect_func import detectTable, parameters

import argparse
#%%
def norm_pdf_page(pdf_file, pg):
    pdf_doc = PdfReader(open(pdf_file, "rb"))
    pdf_page = pdf_doc.pages[pg-1]
    pdf_page.cropbox.upper_left = (0, list(pdf_page.mediabox)[-1])
    pdf_page.cropbox.lower_right = (list(pdf_page.mediabox)[-2], 0)
    print(pdf_page.cropbox.upper_left,pdf_page.cropbox.lower_right)
    return pdf_page

def pdf_page2img(pdf_file, pg, save_image=True):
    img_page = convert_from_path(pdf_file, first_page=pg, last_page=pg,poppler_path=r"C:\Program Files (x86)\Poppler\poppler-23.10.0\Library\bin")[0] #'C:\Program Files (x86)\poppler-23.11.0\Library\bin')[0]
    if save_image:
        img=pdf_file[:-4]+"-"+str(pg)+".jpg"
        img_page.save(img)
    return np.array(img_page)

def outpout_yolo(output):
    output=output.split("\n")
    output.remove("")

    bboxes=[]
    for x in output:
        cleaned_output=x.split(" ")
        cleaned_output.remove("")
        cleaned_output=[eval(x) for x in cleaned_output]
        bboxes.append(cleaned_output)
    
    return bboxes

def img_dim(img, bbox):
    H_img,W_img,_=img.shape
    x1_img, y1_img, x2_img, y2_img,_,_=bbox
    w_table, h_table=x2_img-x1_img, y2_img-y1_img
    return [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]

def norm_bbox(img, bbox, x_corr=0.05, y_corr=0.05):
    [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]=img_dim(img, bbox)
    x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm=x1_img/W_img, y1_img/H_img, x2_img/W_img, y2_img/H_img
    w_img_norm, h_img_norm=w_table/W_img, h_table/H_img
    w_corr=w_img_norm*x_corr
    h_corr=h_img_norm*x_corr

    return [x1_img_norm-w_corr,y1_img_norm-h_corr/2,x2_img_norm+w_corr,y2_img_norm+2*h_corr]


def bboxes_pdf(img, pdf_page, bbox, save_cropped=True):
    W_pdf = float(pdf_page.cropbox.lower_right[0])
    H_pdf = float(pdf_page.cropbox.upper_left[1])

    [x1_img_norm, y1_img_norm, x2_img_norm, y2_img_norm] = norm_bbox(img, bbox)

    # Calculate x and y coordinates using tuple elements
    x1, y1 = x1_img_norm * W_pdf, (1 - y1_img_norm) * H_pdf
    x2, y2 = x2_img_norm * W_pdf, (1 - y2_img_norm) * H_pdf


    
    if save_cropped:
        page=copy.copy(pdf_page)
        page.cropbox.upperLeft = (x1, y1)
        page.cropbox.lowerRight = (x2, y2)
        output = PdfFileWriter()
        output.addPage(page)

        with open("cropped_"+pdf_file[:-4]+"-"+str(pg)+".pdf", "wb") as out_f:
            output.write(out_f)
    

    return [x1, y1, x2, y2]

#%%
def detect_tables(pdf_path,page):
    pdf_file=pdf_path
    pg=page

    see_example=True
    img_path=pdf_file[:-4]+"-"+str(pg)+".jpg"
    pdf_page=norm_pdf_page(pdf_file, pg)
    img = pdf_page2img(pdf_file, pg, save_image=True)

    opt=parameters(img_path)
    output_detect=detectTable(opt)
    output=outpout_yolo(output_detect)


    os.remove(img_path)
    os.rmdir("outputs")

    if see_example:
            for i,out in enumerate(output):
                [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]=img_dim(img, out)
               
                expand_by = 60  # This value can be adjusted according to your needs
                x1_img_expanded = max(x1_img - expand_by, 0)
                y1_img_expanded = max(y1_img - expand_by, 0)
                x2_img_expanded = min(x2_img + expand_by, W_img)
                y2_img_expanded = min(y2_img + expand_by, H_img)
    
                cropped_img_array = img[y1_img_expanded:y2_img_expanded, x1_img_expanded:x2_img_expanded]
                cropped_img = Image.fromarray(cropped_img_array)
                cropped_img.save(pdf_file[:-4] + "-" + str(pg) + "-" + str(i) + ".png")
           


pdf_file_path = r"sample.pdf" 
with open(pdf_file_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_length= len(pdf_reader.pages)
for i in range(1,pdf_length+1):
  detect_tables(pdf_file_path,i)




# %%
