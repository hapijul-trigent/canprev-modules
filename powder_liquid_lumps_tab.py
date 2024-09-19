import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tools, torch, webcolors, colorsys, tempfile


# model_powder_liquid_lump = tools.load_yolo_model(model_path='models/model_powder_liquid_lump.pt')
# Update webcolors with custom color map
# webcolors._definitions._CSS3_HEX_TO_NAMES.update(CUSTOM_CSS3_COLORS_HEX_MAP)

# Function to get the closest color name
def get_color_name_by_rgb(rgb):
    try:
        hex_color = webcolors.rgb_to_hex(rgb)
        return hex_color, webcolors.hex_to_name(hex_color)
    except ValueError:
        closest_color = min(
            webcolors._definitions._CSS3_HEX_TO_NAMES.items(),
            key=lambda x: sum((v - c) ** 2 for v, c in zip(webcolors.hex_to_rgb(x[0]), rgb))
        )
        return closest_color

# Function to detect the dominant color in HSV using k-means clustering
def get_dominant_color_hsv(image, k=4):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    resized_image = cv2.resize(blurred_image, (28, 28), interpolation=cv2.INTER_AREA)

    # st.image(resized_image, channels='BGR')
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    pixels = hsv_image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    dominant_color = np.round(dominant_color).astype(int)

    print(f'Dominant color HSV: {dominant_color}')
    return dominant_color

def img_process(pill_image_path):
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/model_powder_liquid_lump.pt', force_reload=True)
    
    # Read the image
    img_cv2 = cv2.imread(pill_image_path)
    if img_cv2 is None:
        raise ValueError("Image could not be loaded. Please check the image path and file format.")

    # Run inference on the image
    results = model(img_cv2)
    df = results.pandas().xyxy[0]

    
    lumps_count = 0
    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']
        print(label)
        if label == 'lumps': lumps_count += 1
        
       
        roi = img_cv2[y1:y2, x1:x2]
        dominant_color = get_dominant_color_hsv(roi)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        
        rgb = colorsys.hsv_to_rgb(dominant_color[0] / 255, dominant_color[1] / 255, dominant_color[2] / 255)
        color_name = get_color_name_by_rgb((int(rgb[0] *255), int(rgb[1]*255), int(rgb[2]*255)))

        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 5)

        
        if label != 'lumps':
            label_text = f'{label} ({confidence:.2f})'
            cv2.putText(img_cv2, label_text, (x1+30, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (200, 0, 0), 5, cv2.LINE_AA)  # Blue label with larger font

           
            color_text = f'Color: {color_name[1]}'
            cv2.putText(img_cv2, color_text, (x1+30, y1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (200, 0, 0), 5, cv2.LINE_AA)  # Blue text with larger font
            # cv2.putText(img_cv2, color_text, (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5, cv2.LINE_AA)  # Green text with thickness 5

        # # Draw the center of the object
        # center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        # cv2.circle(img_cv2, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)  # Red center dot

    # Resize image if too large for display
    img_height, img_width = img_cv2.shape[:2]
    max_display_height = 800
    if img_height > max_display_height:
        scale_factor = max_display_height / img_height
        img_cv2 = cv2.resize(img_cv2, None, fx=scale_factor, fy=scale_factor)

    return img_cv2, lumps_count

def panel():

    uploaded_file = st.file_uploader("Upload a pill image", type=["jpg", "jpeg", "png"], key='PowderLiquidLumps')

    if uploaded_file is not None:
        
        imageORG, imageAnnotated = st.columns([1,1], gap='large')
        with imageORG:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        annotated_img, lumps_count = img_process(temp_file_path)
        
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil_image = Image.fromarray(annotated_img_rgb)
        
        with imageAnnotated:
            st.image(annotated_pil_image, caption=f'Total Lumps: {lumps_count}', use_column_width=True)

