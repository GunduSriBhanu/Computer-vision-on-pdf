import boto3
import trp
import trp.trp2 as t2
# Textract Caller
from textractcaller.t_call import call_textract, Textract_Features
# Textract Response Parser
from trp import Document

import pandas as pd
import json
from pandas import json_normalize
from textractcaller import call_textract, Textract_Features
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import Pretty_Print_Table_Format, Textract_Pretty_Print, get_string, get_tables_string
from trp import Document
from trp.trp2 import TDocument, TDocumentSchema
from trp.t_pipeline import order_blocks_by_geo
#from IPython.display import display
#import pandas as pd
from pdf2image import convert_from_path
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pandas as pd
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display

# Specify the AWS credentials and region
aws_access_key_id = ''
aws_secret_access_key = ''
region_name = ''  # Specify your desired region
# Example usage:
bucket_name = "demo-bucket"
object_name = "sample.pdf"

path = r"sample.pdf"
doc = convert_from_path(path)


def generate_s3_uri(bucket_name, object_name):
    return f"s3://{bucket_name}/{object_name}"

# Create a Textract client with the specified credentials and region
client = boto3.client(
    'textract',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


documentName = generate_s3_uri(bucket_name, object_name) 
textract_json = call_textract(input_document=documentName, features = [Textract_Features.TABLES], boto3_textract_client = client)

textract_json_layout = call_textract(input_document=documentName, features = [Textract_Features.LAYOUT], boto3_textract_client = client)

# Get the "Blocks" list from the dictionary
blocks = textract_json_layout.get('Blocks', [])

# Initialize a set to store unique BlockType values
unique_block_types = set()

# Iterate through each block and extract the BlockType
for block in blocks:
    block_type = block.get('BlockType')
    if block_type:
        unique_block_types.add(block_type)

# Print unique values of BlockType
#print(unique_block_types)

# Filter blocks for specific block types
filtered_blocks = [block for block in textract_json_layout.get('Blocks', []) if block.get('BlockType') in ['LAYOUT_FIGURE']]

# Convert filtered_blocks to DataFrame
df = json_normalize(filtered_blocks)

#print(df)
def convert_pdf_to_images(bucket_name,object_name):
    page_images = []
    page_dimensions = []
    
    # Create a Boto3 client for S3
    s3 = boto3.client('s3',region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
    )

    # Download the PDF file from the S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key=object_name)
    pdf_bytes = response['Body'].read()

    # Convert PDF bytes to a list of PIL images
    pdf_images = convert_from_bytes(pdf_bytes)
   

    for i, pdf_image in enumerate(pdf_images):
        # Convert PIL image to OpenCV format (BGR)
        img_array = cv2.cvtColor(np.array(pdf_image), cv2.COLOR_RGB2BGR)

        # Get the dimensions of the image
        height, width, _ = img_array.shape

        # Append the image and its dimensions to the lists
        page_images.append(img_array)
        page_dimensions.append((width, height))

    return page_images, page_dimensions



page_images, page_dimensions = convert_pdf_to_images(bucket_name, object_name)


page_dict = {}

# Populate the dictionary
for i, (width, height) in enumerate(page_dimensions):
    page_dict[i + 1] = (width, height)

# Function to get original image dimensions
def get_orig_img_dimensions(page_num):
    return page_dict.get(page_num, None)

# Create 'orig_img' column
df['orig_img'] = df['Page'].apply(get_orig_img_dimensions)

# Split 'orig_img' into two columns
df[['origimg_width', 'origimg_height']] = df['orig_img'].apply(lambda x: pd.Series(x))

# Drop the original 'orig_img' column
df.drop(columns=['orig_img'], inplace=True)

df['Left'] = df['Geometry.BoundingBox.Left'] * df['origimg_width']
df['Top'] = df['Geometry.BoundingBox.Top'] * df['origimg_height']
df['bbox_width'] = df['Geometry.BoundingBox.Width'] * df['origimg_width']
df['bbox_height'] = df['Geometry.BoundingBox.Height'] * df['origimg_height']


df.to_csv("Geometry_Figures.csv")

# EXTRACTION 

df_geom = pd.read_csv(r"C:\Users\SGundu\Textract-Research\Geometry_Figures.csv")



page = doc[7]
page_number = 1

original_img = cv2.cvtColor(np.asarray(page), code=cv2.COLOR_RGB2BGR)


# Convert NumPy array back to PIL Image
pil_img = Image.fromarray(original_img)
img_width, img_height = pil_img.size
draw = ImageDraw.Draw(pil_img)
print("Image size is:", img_width, img_height)


WIDTH = original_img.shape[1]
HEIGTH = original_img.shape[0]
print(WIDTH, HEIGTH)
LEFT_PADDING = 15
RIGHT_PADDING = 5
TOP_PADDING = 5
BOTTOM_PADDING = 65

left = df['Left'][25]
top = df['Top'][25]
width =  df['bbox_width'][25]
height = df['bbox_height'][25]

print(left, top, width, height)
font_size=10
line_width=5

points = ((left,top),
          (left + width, top),
          (left + width, top + height),
          (left , top + height),
          (left, top))

#Draw bounding box and label text
draw.line(points, fill="limegreen", width=line_width)
#print(draw.line(points, fill="limegreen", width=line_width))
pil_img.save("orig_img.png")
x = df_geom['Geometry.BoundingBox.Width'][25]
y = df_geom['Geometry.BoundingBox.Height'][25]
w = df_geom['Geometry.BoundingBox.Left'][25]
h = df_geom['Geometry.BoundingBox.Top'][25]
cropped_img = pil_img.crop((
    x * img_width,
    y * img_height,
    (x + w) * img_width,
    (y + h) * img_height
))

#cropped_img.show()
# Find the minimum and maximum x and y coordinates
min_x = min(point[0] for point in points)
min_y = min(point[1] for point in points)
max_x = max(point[0] for point in points)
max_y = max(point[1] for point in points)

# Crop the image using the bounding box coordinates
crop_img = pil_img.crop((min_x, min_y, max_x, max_y))

# Save or display the cropped image
crop_img.save("cropped_image.png")
