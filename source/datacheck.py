import cv2
import pytesseract
from PIL import Image
import numpy as np
import csv

pytesseract.pytesseract.tesseract_cmd = 'C:/OCR/Tesseract-OCR/tesseract.exe'

# Function to convert the image to grayscale
def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to convert the grayscale image to CSV
def convert_to_csv(gray_image_path, output_csv_path):
    gray_image = cv2.imread(gray_image_path, 0)
    gray_image = cv2.bitwise_not(gray_image)
    threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    threshold_image = cv2.erode(threshold_image, None, iterations=2)
    threshold_image = cv2.dilate(threshold_image, None, iterations=2)

    image = Image.fromarray(threshold_image)
    text = pytesseract.image_to_string(image)

    with open(output_csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Character', 'Count'])
        for char, count in text.items():
            writer.writerow([char, count])

# Path to the input PNG image
input_png_image_path = 'path/to/your/input.png'

# Path to the output CSV file
output_csv_path = 'path/to/your/output.csv'

# Convert the input image to grayscale
grayscale_image_path = 'grayscale_image.png'
cv2.imwrite(grayscale_image_path, convert_to_grayscale(input_png_image_path))

# Convert the grayscale image to CSV
convert_to_csv(grayscale_image_path, output_csv_path)