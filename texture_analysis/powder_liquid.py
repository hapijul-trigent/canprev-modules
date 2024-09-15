import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import streamlit as st

# Load the YOLOv5 model
@st.cache_resource(show_spinner=False)
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='bestPowderLiquid.pt', force_reload=True)
if __name__ == '__main__':
    model = load_model()

def get_rois(image_np, model, border_fraction=0.03):
    # Perform inference
    with torch.no_grad():
        results = model(image_np)
    
    # Get predictions (bounding boxes, confidence, class)
    predictions = results.xyxy[0].cpu().numpy()
    
    rois = []
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        roi = image_np[y1:y2, x1:x2]
        
        # Get the dimensions of the ROI
        height, width = roi.shape[:2]
        
        # Calculate the border size
        border_size_x = int(width * border_fraction)
        border_size_y = int(height * border_fraction)
        
        # Define new coordinates for the cropped ROI
        new_x1 = border_size_x
        new_y1 = border_size_y
        new_x2 = width - border_size_x
        new_y2 = height - border_size_y
        
        # Check bounds to avoid errors
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(width, new_x2)
        new_y2 = min(height, new_y2)
        
        # Crop the ROI from the center with the border left
        cropped_roi = roi[new_y1:new_y2, new_x1:new_x2]
        
        rois.append(cropped_roi)
    
    return rois, predictions


def analyze_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Color Histogram
    hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
    fig, ax = plt.subplots()
    ax.plot(hue_hist)
    ax.set_title('Hue Histogram')
    ax.set_xlabel('Hue Value')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    fig, ax = plt.subplots()
    ax.imshow(edges, cmap='gray')
    ax.set_title('Edge Detection')
    ax.axis('off')
    st.pyplot(fig)
    
    # Thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    fig, ax = plt.subplots()
    ax.imshow(thresh, cmap='gray')
    ax.set_title('Thresholding for Foreign Particles')
    ax.axis('off')
    st.pyplot(fig)
    
    # Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = roi.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax.set_title('Contour Detection of Foreign Particles')
    ax.axis('off')
    st.pyplot(fig)
    
    # Texture Entropy
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    texture_entropy = entropy(histogram)
    st.write(f'Texture Entropy: {texture_entropy:.2f}')
    
    # Local Binary Pattern (LBP)
    P = 8
    R = 1
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    fig, ax = plt.subplots()
    ax.imshow(lbp, cmap='gray')
    ax.set_title('Local Binary Pattern (LBP)')
    ax.axis('off')
    st.pyplot(fig)
    
    # Gabor Filter
    def gabor_filter(image, ksize=31, sigma=4.0, theta=0.0, lambd=10.0, gamma=0.5, phi=0.0):
        gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
        return cv2.filter2D(image, cv2.CV_8UC3, gabor)

    gabor_images = []
    for theta in range(4):
        theta = theta / 4.0 * np.pi
        filtered_image = gabor_filter(gray, ksize=31, sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, phi=0.0)
        gabor_images.append(filtered_image)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    for i, img in enumerate(gabor_images):
        axs[i // 2, i % 2].imshow(img, cmap='gray')
        axs[i // 2, i % 2].set_title(f'Gabor Filter Theta={i * 45}°')
        axs[i // 2, i % 2].axis('off')
    
    st.pyplot(fig)

# # Streamlit app layout
# st.title("YOLOv5 Object Detection and Analysis")

# # File uploader
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert the file to a format OpenCV can read
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
    
#     # Get ROIs from model
#     rois, predictions = get_rois(image_np, model)
    
#     # Display the image with bounding boxes
#     annotated_image = image_np.copy()
#     for pred in predictions:
#         x1, y1, x2, y2, conf, cls = pred
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(annotated_image, f"{model.names[int(cls)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     st.image(annotated_image, caption="Detected Objects", use_column_width=True)
    
#     # Analyze each detected ROI
#     for roi in rois:
#         if roi.size == 0:
#             continue
#         analyze_roi(roi)
