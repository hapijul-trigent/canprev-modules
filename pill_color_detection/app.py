import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import colorsys
from .settings import CUSTOM_CSS3_COLORS_HEX_MAP
import webcolors

webcolors._definitions._CSS3_HEX_TO_NAMES.update(CUSTOM_CSS3_COLORS_HEX_MAP)


# Fetch Color HEX
def get_color_name_by_rgb(rgb):
    """
    Convert RGB to the closest known web color name.

    :param rgb: Tuple containing Red, Green, Blue (R, G, B) values (each 0-255)
    :return: Closest color name, exact color name if found, or 'Unknown'
    """
    try:
        # Get the exact color name if it matches
        hex_color = webcolors.rgb_to_hex(rgb)
        return hex_color, webcolors.hex_to_name(hex_color)
    except ValueError:
        # If no exact match is found, find the closest color name
        closest_color = min(webcolors._definitions._CSS3_HEX_TO_NAMES.items(),
                            key=lambda x: sum((v - c) ** 2 for v, c in zip(webcolors.hex_to_rgb(x[0]), rgb)))
        return closest_color



def crop_center(image, crop_width, crop_height):
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate center coordinates
    centerX, centerY = w // 2, h // 2
    
    # Calculate the cropping box coordinates
    startX = centerX - crop_width // 2
    startY = centerY - crop_height // 2
    endX = centerX + crop_width // 2
    endY = centerY + crop_height // 2
    
    # Make sure the coordinates are within the image bounds
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w, endX)
    endY = min(h, endY)
    
    # Crop the image
    cropped_image = image[startY:endY, startX:endX]
    
    return cropped_image

# Function to detect the dominant color using k-means clustering
def get_dominant_color_hsv(image, k=4):
    # Resize the image to reduce computation
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    resized_image = cv2.resize(blurred_image, (28, 28), interpolation=cv2.INTER_AREA)


    # st.image(resized_image, channels='BGR')
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Reshape the image to be a list of pixels
    pixels = hsv_image.reshape((-1, 3))

    # Convert to float type for k-means clustering
    pixels = np.float32(pixels)

    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get the most frequent cluster center
    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]

    # Round the HSV values to the nearest integer
    dominant_color = np.round(dominant_color).astype(int)

    # Print and return dominant color in HSV
    print(f'Dominant color HSV: {dominant_color}')

    return dominant_color



def process_image(pill_image_path):
    # Load the YOLOv5 model
    model = torch.hub.load('pill_color_detection/yolov5', 'custom', path='pill_color_detection/best.pt', source='local')

    # Load the image using OpenCV
    img_cv2 = cv2.imread(pill_image_path)
    if img_cv2 is None:
        raise ValueError("Image could not be loaded. Please check the image path and file format.")

    # Perform inference using YOLOv5 directly on the OpenCV image
    results = model(img_cv2)

    df = results.pandas().xyxy[0]

    shape_mapping = {
        0: 'Oval_Capsule',
        1: 'Oblong_Capsule',
        2: 'Round_Capsule',
        3: 'Oval_Pill',
        4: 'Rectangle_Pill',
        5: 'Round_Pill',
        6: 'Square_Pill',
        7: 'Triangle_Pill',
        8: 'Desiccant',
        9: 'Scoop'
    }

    scope_present = False
    desiccant_present = False

    # Prepare an array to store ROIs for visualization
    roi_images = []

    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        roi = img_cv2[y1:y2, x1:x2]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.rectangle(mask, (0, 0), (roi.shape[1], roi.shape[0]), 255, -1)

        masked_roi = cv2.bitwise_and(hsv_roi, hsv_roi, mask=mask)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        shape_id = int(row['class'])
        shape_label = shape_mapping.get(shape_id, 'Unknown')

        if shape_label == 'Scoop':
            scope_present = True
        if shape_label == 'Desiccant':
            desiccant_present = True

        cv2.circle(img_cv2, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)

        if shape_label not in ['Scoop', 'Desiccant']:
            dominant_color = get_dominant_color_hsv(roi)
            print('HSV: ', dominant_color[0], dominant_color[1], dominant_color[2])

            # Get closest color name based on HSV
            # color_name = closest_color_hsv(dominant_color)
            rgb = colorsys.hsv_to_rgb(dominant_color[0]/255, dominant_color[1]/255, dominant_color[2]/255)
            print('HSv to RGB : ', int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            color_name = get_color_name_by_rgb((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
            print(color_name)

            font_scale_shape = 2
            thickness_shape = 8
            cv2.putText(img_cv2, f'{shape_label}', (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_shape, (0, 255, 0), thickness_shape)  # White
            cv2.putText(img_cv2, f'Color: {color_name[1]}', (center_x + 10, center_y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale_shape, (0, 255, 0), thickness_shape)  # White

        else:
            font_scale_shape = 2.5
            thickness_shape = 6

            cv2.putText(img_cv2, f'{shape_label}', (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_shape, (0, 255, 0), thickness_shape)

        # Store the ROI image for visualization
        roi_images.append(roi)

    total_pills = len(df[df['class'] < 8])

    # Get image dimensions
    img_height, img_width = img_cv2.shape[:2]

    # Adjust text size based on image dimensions
    font_scale = max(2.0, img_height / 400)  # Adjusted font scale for readability
    thickness = max(4, img_height // 200)    # Adjusted thickness for readability

    # Adjusting the text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 255)  # Red color for readability

    text = f'Total pills: {total_pills}'
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img_width - text_size[0]) // 2
    text_y = text_size[1] + int(20 * font_scale)  # Place text at the top center

    # Ensure the image fits within display area by resizing if necessary
    max_display_height = 800
    if img_height > max_display_height:
        scale_factor = max_display_height / img_height
        img_cv2 = cv2.resize(img_cv2, None, fx=scale_factor, fy=scale_factor)
        img_height, img_width = img_cv2.shape[:2]
        text_size = (int(text_size[0] * scale_factor), int(text_size[1] * scale_factor))
        text_x = (img_width - text_size[0]) // 2
        text_y = text_size[1] + int(20 * font_scale * scale_factor)


    # cv2.putText(img_cv2, text, (text_x, text_y), font, font_scale, text_color, thickness)

    # Desiccant present annotation
    if desiccant_present:
        desiccant_text = 'Desiccant Present'
        desiccant_y = text_y + int(60 * font_scale)  # Adjusted position for better alignment
        cv2.putText(img_cv2, desiccant_text, (text_x, desiccant_y), font, font_scale, text_color, thickness)

    # Scope present annotation
    if scope_present:
        scope_text = 'Scope Present'
        scope_y = desiccant_y + int(60 * font_scale)  # Adjusted position for better alignment
        cv2.putText(img_cv2, scope_text, (text_x, scope_y), font, font_scale, text_color, thickness)

    return img_cv2, total_pills


def main():

    uploaded_file = st.file_uploader("Upload a pill image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        imageORG, imageAnnotated = st.columns([1,1], gap='large')
        with imageORG:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        annotated_img, total_pills = process_image(temp_file_path)
        
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil_image = Image.fromarray(annotated_img_rgb)
        
        with imageAnnotated:
            st.image(annotated_pil_image, caption=f'Total Pills: {total_pills}', use_column_width=True)
