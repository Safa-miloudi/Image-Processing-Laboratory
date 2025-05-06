import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import requests
import tempfile
# ================================================
# TP1 - Histogram Analysis Functions
# ================================================

def convert_to_gray(image_np):
    """Convert image to grayscale"""
    return np.dot(image_np[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

def calculate_histograms(img):
    """Calculate histograms for all channels"""
    if len(img.shape) == 2:  # Grayscale
        hist = np.histogram(img, bins=256, range=(0, 256))[0]
        return [hist]
    else:  # Color image (3 channels)
        return [np.histogram(img[:,:,i], bins=256, range=(0, 256))[0] for i in range(3)]

def normalize_histograms(hists, pixel_count):
    """Normalize histograms"""
    return [hist / pixel_count for hist in hists]

def calculate_cumulative(hists):
    """Calculate cumulative histograms"""
    return [np.cumsum(hist) for hist in hists]

def plot_combined_histograms(hists, norm_hists, cum_hists, title, colors, labels):
    """Plot combined histograms (raw, normalized, cumulative)"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    for i, (hist, norm_hist, cum_hist) in enumerate(zip(hists, norm_hists, cum_hists)):
        axes[0].plot(hist, color=colors[i], label=labels[i], alpha=0.7)
        axes[1].plot(norm_hist, color=colors[i], label=labels[i], alpha=0.7)
        axes[2].plot(cum_hist, color=colors[i], label=labels[i], alpha=0.7)
    
    for ax, suffix in zip(axes, ["Raw", "Normalized", "Cumulative"]):
        ax.set_title(f"{suffix} Histogram - {title}")
        ax.grid(True)
        if len(hists) > 1:
            ax.legend()
    
    plt.tight_layout()
    return fig

def show_grayscale_histograms(gray_img):
    """Show histograms for grayscale image"""
    hists = calculate_histograms(gray_img)
    norm_hists = normalize_histograms(hists, gray_img.shape[0] * gray_img.shape[1])
    cum_hists = calculate_cumulative(norm_hists)
    return plot_combined_histograms(hists, norm_hists, cum_hists, 
                                  "Grayscale", ["gray"], ["Intensity"])

def show_color_space_histograms(img, color_space):
    """Show histograms for color spaces"""
    if color_space == "RGB":
        converted_img = img
        colors = ["red", "green", "blue"]
        labels = ["Red", "Green", "Blue"]
    elif color_space == "HSV":
        converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        colors = ["magenta", "cyan", "yellow"]
        labels = ["Hue", "Saturation", "Value"]
    elif color_space == "LAB":
        converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        colors = ["gray", "green", "blue"]
        labels = ["Luminance", "A", "B"]
    
    hists = calculate_histograms(converted_img)
    norm_hists = normalize_histograms(hists, img.shape[0] * img.shape[1])
    cum_hists = calculate_cumulative(norm_hists)
    
    return plot_combined_histograms(hists, norm_hists, cum_hists, 
                                  color_space, colors, labels)

# ================================================
# TP2 - Histogram Transformation Functions
# ================================================

def histogram_translation(img, offset):
    """Translate histogram by adding an offset"""
    if len(img.shape) == 3:  # Color image
        translated = np.clip(img.astype(int) + offset, 0, 255).astype(np.uint8)
    else:  # Grayscale
        translated = np.clip(img.astype(int) + offset, 0, 255).astype(np.uint8)
    return translated

def histogram_inversion(img):
    """Invert the histogram (negative image)"""
    return 255 - img

def dynamic_expansion(img):
    """Expand dynamic range to [0, 255]"""
    if len(img.shape) == 3:  # Color image
        expanded = np.zeros_like(img)
        for i in range(3):
            channel = img[:,:,i]
            min_val, max_val = np.min(channel), np.max(channel)
            if min_val == max_val:
                expanded[:,:,i] = channel
            else:
                expanded[:,:,i] = ((channel - min_val) * (255 / (max_val - min_val))).astype(np.uint8)
    else:  # Grayscale
        min_val, max_val = np.min(img), np.max(img)
        if min_val == max_val:
            return img
        expanded = ((img - min_val) * (255 / (max_val - min_val))).astype(np.uint8)
    return expanded

def histogram_equalization(img):
    """Perform histogram equalization"""
    if len(img.shape) == 2:  # Grayscale
        return cv2.equalizeHist(img)
    else:  # Color (apply to each channel)
        channels = [cv2.equalizeHist(img[:,:,i]) for i in range(3)]
        return np.stack(channels, axis=-1)

def plot_transformation_results(original_img, transformed_img, title):
    """Plot original and transformed images with histograms"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    if len(original_img.shape) == 2:
        axes[0,0].imshow(original_img, cmap='gray')
    else:
        axes[0,0].imshow(original_img)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    
    # Transformed image
    if len(transformed_img.shape) == 2:
        axes[0,1].imshow(transformed_img, cmap='gray')
    else:
        axes[0,1].imshow(transformed_img)
    axes[0,1].set_title(title)
    axes[0,1].axis('off')
    
    # Original histogram
    if len(original_img.shape) == 2:
        axes[1,0].hist(original_img.ravel(), 256, [0,256], color='gray')
    else:
        for i, color in enumerate(['r','g','b']):
            axes[1,0].hist(original_img[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7)
    axes[1,0].set_title("Original Histogram")
    
    # Transformed histogram
    if len(transformed_img.shape) == 2:
        axes[1,1].hist(transformed_img.ravel(), 256, [0,256], color='gray')
    else:
        for i, color in enumerate(['r','g','b']):
            axes[1,1].hist(transformed_img[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7)
    axes[1,1].set_title(f"{title} Histogram")
    
    plt.tight_layout()
    return fig
# ================================================
# TP3 - Color Quantization and Histogram Functions
# ================================================
def color_quantization(img, k):
    """Perform color quantization using k-means clustering"""
    # Reshape the image to be a list of pixels
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Flatten the labels array
    labels = labels.flatten()
    
    # Convert all pixels to the color of the centroids
    quantized_img = centers[labels]
    
    # Reshape back to the original image dimension
    quantized_img = quantized_img.reshape(img.shape)
    
    return quantized_img, centers

def calculate_quantized_histogram(quantized_img, centers):
    """Calculate histogram and normalized histogram of quantized colors"""
    # Reshape image to list of pixels
    pixels = quantized_img.reshape((-1, 3))
    total_pixels = pixels.shape[0]
    
    # Convert to tuples so we can count unique colors
    pixels = [tuple(pixel) for pixel in pixels]
    
    # Count occurrences of each color
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Create a full histogram with all possible colors (initialized to 0)
    k = centers.shape[0]
    hist = np.zeros(k)
    norm_hist = np.zeros(k)
    
    # Map each unique color to its cluster center
    for color, count in zip(unique_colors, counts):
        # Find which cluster center this color corresponds to
        distances = np.linalg.norm(centers - color, axis=1)
        cluster_idx = np.argmin(distances)
        
        if cluster_idx < k:
            hist[cluster_idx] = count
            norm_hist[cluster_idx] = count / total_pixels
    
    return hist, norm_hist

def plot_quantization_results(original_img, quantized_img, k, hist, norm_hist):
    """Plot original, quantized images and histograms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0,0].imshow(original_img)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    
    # Quantized image
    axes[0,1].imshow(quantized_img)
    axes[0,1].set_title(f"Quantized Image (k={k})")
    axes[0,1].axis('off')
    
    # Quantized histogram
    axes[1,0].bar(range(k), hist, color='blue', alpha=0.7)
    axes[1,0].set_title(f"Quantized Color Histogram (k={k})")
    axes[1,0].set_xlabel("Color Cluster Index")
    axes[1,0].set_ylabel("Frequency")
    axes[1,0].grid(True)
    
    # Normalized quantized histogram
    axes[1,1].bar(range(k), norm_hist, color='green', alpha=0.7)
    axes[1,1].set_title(f"Normalized Quantized Histogram (k={k})")
    axes[1,1].set_xlabel("Color Cluster Index")
    axes[1,1].set_ylabel("Normalized Frequency")
    axes[1,1].grid(True)
    
    plt.tight_layout()
    return fig

# ================================================
# TP4 - Convolution Filters
# ================================================

def apply_filter(image, filter_type, kernel_size=3):
    """Apply various convolution filters to the image"""
    if len(image.shape) == 3:  # Color image
        # Convert to grayscale for some filters that work better on single channel
        if filter_type in ['sobel', 'prewitt', 'roberts']:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Define all the filters
    filters = {
        # Average filters
        'average_simple': np.ones((kernel_size, kernel_size)) / (kernel_size**2),
        'average_weighted': np.array([[2,4,2], [4,2,4], [2,4,2]]) / 24,
        
        # Gaussian filters
        'gaussian_3x3': np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16,
        'gaussian_5x5': cv2.getGaussianKernel(5, 0) @ cv2.getGaussianKernel(5, 0).T,
        
        # High-pass filters
        'highpass_1': np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]),
        'highpass_2': np.array([[-1,-1,-1], [0,0,0], [1,1,1]]),
        
        # Edge detection filters
        'sobel_x': np.array([[-1,0,1], [-2,0,2], [-1,0,1]]),
        'sobel_y': np.array([[-1,-2,-1], [0,0,0], [1,2,1]]),
        'prewitt_x': np.array([[-1,0,1], [-1,0,1], [-1,0,1]]),
        'prewitt_y': np.array([[-1,-1,-1], [0,0,0], [1,1,1]]),
        'roberts_x': np.array([[0,1], [-1,0]]),
        'roberts_y': np.array([[1,0], [0,-1]]),
    }
    
    # Special cases (non-linear filters)
    if filter_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'min':
        return cv2.erode(image, np.ones((kernel_size, kernel_size)))
    elif filter_type == 'max':
        return cv2.dilate(image, np.ones((kernel_size, kernel_size)))
    elif filter_type == 'geometric_mean':
        # Geometric mean is not directly available in OpenCV
        # We'll implement a simplified version
        if len(image.shape) == 2:
            image = np.float32(image)
            result = np.exp(cv2.boxFilter(np.log(image + 1e-10), -1, (kernel_size, kernel_size)))
            return np.uint8(np.clip(result, 0, 255))
        else:
            channels = [np.exp(cv2.boxFilter(np.log(ch + 1e-10), -1, (kernel_size, kernel_size))) 
                       for ch in cv2.split(np.float32(image))]
            return np.uint8(np.clip(cv2.merge(channels), 0, 255))
    else:
        kernel = filters.get(filter_type)
        if kernel is None:
            return image
        
        # Apply the convolution
        if len(image.shape) == 2:  # Grayscale
            return cv2.filter2D(image, -1, kernel)
        else:  # Color
            channels = [cv2.filter2D(ch, -1, kernel) for ch in cv2.split(image)]
            return cv2.merge(channels)

def plot_filter_results(original_img, filtered_img, filter_name):
    """Plot original and filtered images with histograms"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    if len(original_img.shape) == 2:
        axes[0,0].imshow(original_img, cmap='gray')
    else:
        axes[0,0].imshow(original_img)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    
    # Filtered image
    if len(filtered_img.shape) == 2:
        axes[0,1].imshow(filtered_img, cmap='gray')
    else:
        axes[0,1].imshow(filtered_img)
    axes[0,1].set_title(filter_name)
    axes[0,1].axis('off')
    
    # Original histogram
    if len(original_img.shape) == 2:
        axes[1,0].hist(original_img.ravel(), 256, [0,256], color='gray')
    else:
        for i, color in enumerate(['r','g','b']):
            axes[1,0].hist(original_img[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7)
    axes[1,0].set_title("Original Histogram")
    
    # Filtered histogram
    if len(filtered_img.shape) == 2:
        axes[1,1].hist(filtered_img.ravel(), 256, [0,256], color='gray')
    else:
        for i, color in enumerate(['r','g','b']):
            axes[1,1].hist(filtered_img[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7)
    axes[1,1].set_title(f"{filter_name} Histogram")
    
    plt.tight_layout()
    return fig
# ================================================
# TP5 - Edge Detection
# ================================================

def apply_sobel(image, color_edge=False):
    """Apply Sobel edge detection"""
    if len(image.shape) == 3 and not color_edge:
        # Convert to grayscale if not doing color edges
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if len(image.shape) == 3 and color_edge:
        # Color edge detection - process each channel separately
        sobel_x = np.zeros_like(image, dtype=np.float32)
        sobel_y = np.zeros_like(image, dtype=np.float32)
        
        for i in range(3):
            channel = image[:,:,i]
            sobel_x[:,:,i] = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y[:,:,i] = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine the gradients
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
    else:
        # Grayscale edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    # Normalize to 0-255
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return gradient_magnitude, gradient_direction

def apply_laplacian(image):
    """Apply Laplacian edge detection"""
    if len(image.shape) == 3:
        # Convert to grayscale for Laplacian
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Take absolute value and convert to 8-bit
    laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return laplacian

def plot_edge_results(original_img, edge_img, magnitude_img=None, direction_img=None, title=""):
    """Plot edge detection results"""
    if magnitude_img is not None and direction_img is not None:
        # Sobel results (magnitude and direction)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        if len(original_img.shape) == 2:
            axes[0,0].imshow(original_img, cmap='gray')
        else:
            axes[0,0].imshow(original_img)
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        
        # Edge image
        axes[0,1].imshow(edge_img, cmap='gray')
        axes[0,1].set_title(f"{title} (Magnitude)")
        axes[0,1].axis('off')
        
        # Gradient magnitude
        axes[1,0].imshow(magnitude_img, cmap='gray')
        axes[1,0].set_title("Gradient Magnitude")
        axes[1,0].axis('off')
        
        # Gradient direction
        axes[1,1].imshow(direction_img, cmap='hsv')
        axes[1,1].set_title("Gradient Direction")
        axes[1,1].axis('off')
    else:
        # Laplacian results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        if len(original_img.shape) == 2:
            axes[0].imshow(original_img, cmap='gray')
        else:
            axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Edge image
        axes[1].imshow(edge_img, cmap='gray')
        axes[1].set_title(title)
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig

# ================================================
# TP6 - Image Segmentation
# ================================================

def simple_threshold(image, threshold_value):
    """Apply simple thresholding"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def otsu_threshold(image):
    """Apply Otsu's thresholding"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def adaptive_threshold(image, block_size=11, C=2):
    """Apply adaptive thresholding"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, block_size, C)
    return binary

def kmeans_segmentation(image, k=2):
    """Apply k-means clustering for segmentation"""
    if len(image.shape) == 3:
        # Reshape to 2D array of pixels
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
    else:
        # For grayscale, convert to 2D array
        pixel_values = image.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    # Flatten the labels array
    labels = labels.flatten()
    
    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels]
    
    # Reshape back to the original image dimension
    if len(image.shape) == 3:
        segmented_image = segmented_image.reshape(image.shape)
    else:
        segmented_image = segmented_image.reshape(image.shape[0], image.shape[1])
    
    return segmented_image, labels, centers

def plot_segmentation_results(original_img, segmented_img, title=""):
    """Plot segmentation results with histograms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    if len(original_img.shape) == 2:
        axes[0,0].imshow(original_img, cmap='gray')
    else:
        axes[0,0].imshow(original_img)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    
    # Segmented image
    if len(segmented_img.shape) == 2:
        axes[0,1].imshow(segmented_img, cmap='gray')
    else:
        axes[0,1].imshow(segmented_img)
    axes[0,1].set_title(title)
    axes[0,1].axis('off')
    
    # Original histogram
    if len(original_img.shape) == 2:
        axes[1,0].hist(original_img.ravel(), 256, [0,256], color='gray')
    else:
        for i, color in enumerate(['r','g','b']):
            axes[1,0].hist(original_img[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7)
    axes[1,0].set_title("Original Histogram")
    axes[1,0].grid(True)
    
    # Segmented histogram
    if len(segmented_img.shape) == 2:
        axes[1,1].hist(segmented_img.ravel(), 256, [0,256], color='blue')
    else:
        for i, color in enumerate(['r','g','b']):
            axes[1,1].hist(segmented_img[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7)
    axes[1,1].set_title(f"{title} Histogram")
    axes[1,1].grid(True)
    
    plt.tight_layout()
    return fig
# ================================================
# TP7 - Connected Components Labeling 
# ================================================

def connected_components_labeling_custom(binary_img):
    """Implement connected components labeling with equivalence table (fixed version)"""
    # Ensure binary image (0 and 255)
    _, binary = cv2.threshold(binary_img, 127, 1, cv2.THRESH_BINARY)
    
    # Initialize labels
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 1
    equivalence = {}
    
    # First pass
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i,j] == 0:
                continue
                
            # Get neighbors (4-connectivity)
            neighbors = []
            if i > 0 and labels[i-1,j] > 0:
                neighbors.append(labels[i-1,j])
            if j > 0 and labels[i,j-1] > 0:
                neighbors.append(labels[i,j-1])
            
            if not neighbors:
                labels[i,j] = current_label
                equivalence[current_label] = current_label
                current_label += 1
            else:
                min_label = min(neighbors)
                labels[i,j] = min_label
                
                # Update equivalence table
                for n in neighbors:
                    if n != min_label:
                        equivalence[n] = min_label
    
    # Resolve equivalence classes (path compression)
    for label in list(equivalence.keys()):
        while equivalence[label] != equivalence.get(equivalence[label], equivalence[label]):
            equivalence[label] = equivalence[equivalence[label]]
    
    # Second pass
    unique_labels = {}
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] > 0:
                labels[i,j] = equivalence.get(labels[i,j], labels[i,j])
                unique_labels[labels[i,j]] = True
    
    # Create color map for visualization
    num_labels = len(unique_labels)
    colors = np.random.randint(0, 255, size=(current_label, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background
    
    colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] < len(colors):  # Safety check
                colored_labels[i,j] = colors[labels[i,j]]
    
    return labels, colored_labels, num_labels

def plot_components_results_custom(original_img, binary_img, colored_labels, num_labels):
    """Plot connected components results from custom implementation"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    if len(original_img.shape) == 2:
        axes[0].imshow(original_img, cmap='gray')
    else:
        axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Binary image
    axes[1].imshow(binary_img, cmap='gray')
    axes[1].set_title("Binary Image")
    axes[1].axis('off')
    
    # Labeled components
    axes[2].imshow(colored_labels)
    axes[2].set_title(f"Connected Components ({num_labels} regions)")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# ================================================
# TP8 - Region Properties Calculation 
# ================================================

def calculate_region_properties(labels, num_labels):
    """Calculate various region properties (fixed version)"""
    properties = []
    
    for label in range(1, num_labels + 1):
        mask = (labels == label).astype(np.uint8)
        
        # Skip if no pixels with this label
        if np.sum(mask) == 0:
            continue
            
        # Calculate moments
        moments = cv2.moments(mask)
        
        # Area
        area = moments['m00']
        if area == 0:
            continue
        
        # Centroid
        centroid_x = moments['m10'] / area
        centroid_y = moments['m01'] / area
        
        # Bounding box
        points = np.argwhere(mask)
        y, x = points[:, 0], points[:, 1]
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # Perimeter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            perimeter = 0
        else:
            perimeter = cv2.arcLength(contours[0], True)
        
        # Compactness
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Eccentricity
        mu20 = moments['mu20'] / area
        mu11 = moments['mu11'] / area
        mu02 = moments['mu02'] / area
        eccentricity = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2) / (mu20 + mu02) if (mu20 + mu02) > 0 else 0
        
        properties.append({
            'label': label,
            'area': area,
            'centroid': (centroid_x, centroid_y),
            'bounding_box': (x_min, y_min, width, height),
            'perimeter': perimeter,
            'compactness': compactness,
            'eccentricity': eccentricity
        })
    
    return properties

def plot_region_properties(original_img, properties, colored_labels):
    """Plot region properties visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with centroids
    if len(original_img.shape) == 2:
        axes[0].imshow(original_img, cmap='gray')
    else:
        axes[0].imshow(original_img)
    
    for prop in properties:
        cx, cy = prop['centroid']
        axes[0].plot(cx, cy, 'r+', markersize=10)
        axes[0].text(cx + 5, cy + 5, f"{prop['label']}", color='red', fontsize=8)
    
    axes[0].set_title("Original Image with Centroids")
    axes[0].axis('off')
    
    # Labeled image with bounding boxes
    axes[1].imshow(colored_labels)
    for prop in properties:
        x, y, w, h = prop['bounding_box']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='white', linewidth=1)
        axes[1].add_patch(rect)
        axes[1].text(x + 5, y + 15, f"{prop['label']}", color='white', fontsize=8)
    
    axes[1].set_title("Labeled Regions with Bounding Boxes")
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig
# ================================================
# TP9 - Video Processing Functions 
# ================================================

def display_video_frames(video_file):
    """Display video frames with playback controls (fixed version)"""
    try:
        # Create a temporary file to store the video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()
        
        # Open the video file
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error opening video file")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        st.write(f"Video Info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds duration")
        
        # Create a slider for frame navigation
        frame_number = st.slider("Select Frame", 0, frame_count-1, 0, key="video_frame_slider")
        
        # Set the video to the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            st.image(frame_rgb, caption=f"Frame {frame_number}", use_container_width=True)
            
            # Display frame information
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Frame {frame_number}/{frame_count-1}")
            with col2:
                st.write(f"Timestamp: {frame_number/fps:.2f} seconds")
            
            # Playback controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⏮ Previous Frame"):
                    frame_number = max(0, frame_number - 1)
                    st.session_state.video_frame_slider = frame_number
                    st.experimental_rerun()
            with col2:
                if st.button("⏯ Play/Pause" if not st.session_state.get('play_video', False) else "⏸ Pause"):
                    st.session_state.play_video = not st.session_state.get('play_video', False)
                    st.experimental_rerun()
            with col3:
                if st.button("⏭ Next Frame"):
                    frame_number = min(frame_count-1, frame_number + 1)
                    st.session_state.video_frame_slider = frame_number
                    st.experimental_rerun()
            
            # Auto-play if play button is pressed
            if st.session_state.get('play_video', False):
                time.sleep(0.5/fps)  # Control playback speed (0.5 = half speed)
                next_frame = (frame_number + 1) % frame_count
                st.session_state.video_frame_slider = next_frame
                st.experimental_rerun()
        
        cap.release()
        
        # Clean up the temporary file
        os.unlink(tfile.name)
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

# ================================================
# Main Application
# ================================================

def main():
    st.title("Image Processing Laboratory")
    
    # Upload image or video
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image or video", 
        type=["jpg", "jpeg", "png", "tiff", "mp4", "avi", "mov"]
    )
    
    if uploaded_file:
        # Check if the file is a video
        if uploaded_file.type.startswith('video'):
            # Create tabs including TP9
            tab_tp9 = st.tabs(["TP9 - Video Processing"])
            
            with tab_tp9:
                st.header("TP9 - Video Processing")
                display_video_frames(uploaded_file)
        else:
            # Process as image (your existing code)
            img_pil = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(img_pil)
            gray_img = convert_to_gray(img_np)
            
            # Create tabs for all TPs (including TP9)
            tab_tp1, tab_tp2, tab_tp3, tab_tp4, tab_tp5, tab_tp6, tab_tp7, tab_tp8, tab_tp9 = st.tabs([
                "TP1 - Histogram Analysis", 
                "TP2 - Histogram Transformations",
                "TP3 - Color Quantization",
                "TP4 - Convolution Filters",
                "TP5 - Edge Detection",
                "TP6 - Image Segmentation",
                "TP7 - Connected Components",
                "TP8 - Region Properties",
                "TP9 - Video Processing"
            ])
        with tab_tp1:
            st.header("TP1 - Image Histogram Analysis")
            
            # Create subtabs for different color spaces
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Original", "Grayscale", "HSV", "LAB", "RGB"
            ])
            with tab1:
                st.image(img_pil, caption="Original Image")

            with tab2:
                st.image(gray_img, caption="Grayscale Image", clamp=True)
                fig = show_grayscale_histograms(gray_img)
                st.pyplot(fig)

            with tab3:
                hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                st.image(hsv_img, caption="HSV Image", channels="BGR")
                fig = show_color_space_histograms(img_np, "HSV")
                st.pyplot(fig)

            with tab4:
                lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                st.image(lab_img, caption="LAB Image", channels="BGR")
                fig = show_color_space_histograms(img_np, "LAB")
                st.pyplot(fig)

            with tab5:
                st.image(img_np, caption="RGB Image")
                fig = show_color_space_histograms(img_np, "RGB")
                st.pyplot(fig)
        
        with tab_tp2:
            st.header("TP2 - Histogram Transformations")
            
            # Choose between color and grayscale processing
            process_mode = st.radio("Processing mode:", 
                                  ["Grayscale", "Color"], 
                                  horizontal=True)
            
            if process_mode == "Grayscale":
                working_img = gray_img
            else:
                working_img = img_np
            
            # Create subtabs for different transformations
            trans_tab1, trans_tab2, trans_tab3, trans_tab4 = st.tabs([
                "Translation", "Inversion", "Dynamic Expansion", "Equalization"
            ])
            
            with trans_tab1:
                st.subheader("Histogram Translation")
                offset = st.slider("Translation offset", -255, 255, 50, key="translation_offset")
                translated_img = histogram_translation(working_img, offset)
                fig = plot_transformation_results(working_img, translated_img, f"Translated (offset={offset})")
                st.pyplot(fig)
            
            with trans_tab2:
                st.subheader("Histogram Inversion")
                inverted_img = histogram_inversion(working_img)
                fig = plot_transformation_results(working_img, inverted_img, "Inverted Image")
                st.pyplot(fig)
            
            with trans_tab3:
                st.subheader("Dynamic Expansion")
                expanded_img = dynamic_expansion(working_img)
                fig = plot_transformation_results(working_img, expanded_img, "Dynamic Expansion")
                st.pyplot(fig)
            
            with trans_tab4:
                st.subheader("Histogram Equalization")
                equalized_img = histogram_equalization(working_img)
                fig = plot_transformation_results(working_img, equalized_img, "Equalized Image")
                st.pyplot(fig)
        with tab_tp3:
            st.header("TP3 - Color Quantization and Histograms")
            
            # Only allow color images for quantization
            if len(img_np.shape) == 2:
                st.warning("Color quantization requires a color image. Please upload a color image.")
            else:
                k = st.slider("Number of colors (k)", 2, 256, 16, key="quantization_k")
                
                st.subheader("Color Quantization Results")
                quantized_img, centers = color_quantization(img_np, k)
                
                st.subheader("Quantized Color Histograms")
                hist, norm_hist = calculate_quantized_histogram(quantized_img, centers)
                
                fig = plot_quantization_results(img_np, quantized_img, k, hist, norm_hist)
                st.pyplot(fig)
                
                # Show the quantized image with container width
                st.image(quantized_img, caption=f"Quantized Image with {k} colors", use_container_width=True)
        with tab_tp4:
            st.header("TP4 - Convolution Filters")
            
            # Choose between color and grayscale processing
            process_mode = st.radio("Processing mode:", 
                                   ["Grayscale", "Color"], 
                                   horizontal=True,
                                   key="filter_mode")
            
            if process_mode == "Grayscale":
                working_img = gray_img
            else:
                working_img = img_np
            
            # Create subtabs for different filter categories
            filter_categories = st.selectbox("Filter Category:", [
                "Average Filters",
                "Gaussian Filters",
                "Median Filter",
                "High-Pass Filters",
                "Edge Detection",
                "Other Filters"
            ])
            
            if filter_categories == "Average Filters":
                filter_type = st.selectbox("Select Average Filter:", [
                    "Simple Average",
                    "Weighted Average"
                ])
                kernel_size = st.slider("Kernel Size (for simple average)", 3, 15, 3, step=2)
                
                if filter_type == "Simple Average":
                    filtered_img = apply_filter(working_img, 'average_simple', kernel_size)
                    filter_name = f"Simple Average ({kernel_size}x{kernel_size})"
                else:
                    filtered_img = apply_filter(working_img, 'average_weighted', 3)
                    filter_name = "Weighted Average (3x3)"
            
            elif filter_categories == "Gaussian Filters":
                filter_type = st.selectbox("Select Gaussian Filter:", [
                    "Gaussian 3x3",
                    "Gaussian 5x5"
                ])
                
                if filter_type == "Gaussian 3x3":
                    filtered_img = apply_filter(working_img, 'gaussian_3x3')
                    filter_name = "Gaussian 3x3"
                else:
                    filtered_img = apply_filter(working_img, 'gaussian_5x5')
                    filter_name = "Gaussian 5x5"
            
            elif filter_categories == "Median Filter":
                kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2)
                filtered_img = apply_filter(working_img, 'median', kernel_size)
                filter_name = f"Median Filter ({kernel_size}x{kernel_size})"
            
            elif filter_categories == "High-Pass Filters":
                filter_type = st.selectbox("Select High-Pass Filter:", [
                    "High-Pass 1 (Edge Enhancement)",
                    "High-Pass 2 (Horizontal Edges)"
                ])
                
                if filter_type == "High-Pass 1 (Edge Enhancement)":
                    filtered_img = apply_filter(working_img, 'highpass_1')
                    filter_name = "High-Pass (Edge Enhancement)"
                else:
                    filtered_img = apply_filter(working_img, 'highpass_2')
                    filter_name = "High-Pass (Horizontal Edges)"
            
            elif filter_categories == "Edge Detection":
                filter_type = st.selectbox("Select Edge Detection Filter:", [
                    "Sobel X",
                    "Sobel Y",
                    "Prewitt X",
                    "Prewitt Y",
                    "Roberts X",
                    "Roberts Y"
                ])
                
                filter_map = {
                    "Sobel X": 'sobel_x',
                    "Sobel Y": 'sobel_y',
                    "Prewitt X": 'prewitt_x',
                    "Prewitt Y": 'prewitt_y',
                    "Roberts X": 'roberts_x',
                    "Roberts Y": 'roberts_y'
                }
                
                filtered_img = apply_filter(working_img, filter_map[filter_type])
                filter_name = f"{filter_type} Edge Detection"
            
            elif filter_categories == "Other Filters":
                filter_type = st.selectbox("Select Other Filter:", [
                    "Minimum",
                    "Maximum",
                    "Geometric Mean"
                ])
                
                kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2)
                
                if filter_type == "Minimum":
                    filtered_img = apply_filter(working_img, 'min', kernel_size)
                    filter_name = f"Minimum Filter ({kernel_size}x{kernel_size})"
                elif filter_type == "Maximum":
                    filtered_img = apply_filter(working_img, 'max', kernel_size)
                    filter_name = f"Maximum Filter ({kernel_size}x{kernel_size})"
                else:
                    filtered_img = apply_filter(working_img, 'geometric_mean', kernel_size)
                    filter_name = f"Geometric Mean ({kernel_size}x{kernel_size})"
            
            # Display results
            st.subheader(filter_name)
            fig = plot_filter_results(working_img, filtered_img, filter_name)
            st.pyplot(fig)
            
            # Show the filtered image
            if len(filtered_img.shape) == 2:
                st.image(filtered_img, caption=filter_name, use_container_width=True, clamp=True)
            else:
                st.image(filtered_img, caption=filter_name, use_container_width=True)
        with tab_tp5:
            st.header("TP5 - Edge Detection")
            
            # Edge detection method selection
            edge_method = st.selectbox("Edge Detection Method:", [
                "Sobel Operator",
                "Laplacian Operator"
            ])
            
            if edge_method == "Sobel Operator":
                # Sobel options
                color_edge = st.checkbox("Color Edge Detection (for Sobel)", value=False)
                
                # Apply Sobel
                magnitude, direction = apply_sobel(img_np, color_edge)
                
                # Create edge image (magnitude thresholded)
                _, edge_img = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
                
                # Plot results
                fig = plot_edge_results(img_np, edge_img, magnitude, direction, "Sobel Edge Detection")
                st.pyplot(fig)
                
                # Show the edge image
                st.image(edge_img, caption="Sobel Edge Image", use_container_width=True, clamp=True)
                               
                
            else:  # Laplacian Operator
                # Apply Laplacian
                laplacian = apply_laplacian(img_np)
                
                # Create edge image (zero-crossing would be better but simple threshold for demo)
                _, edge_img = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
                
                # Plot results
                fig = plot_edge_results(img_np, edge_img, title="Laplacian Edge Detection")
                st.pyplot(fig)
                
                # Show the edge image
                st.image(edge_img, caption="Laplacian Edge Image", use_container_width=True, clamp=True)
        
        with tab_tp6:
            st.header("TP6 - Image Segmentation")
            
            # Segmentation method selection
            seg_method = st.selectbox("Segmentation Method:", [
                "Simple Thresholding",
                "Otsu's Thresholding",
                "Adaptive Thresholding",
                "K-means Clustering"
            ])
            
            if seg_method == "Simple Thresholding":
                st.subheader("Simple Thresholding")
                threshold_value = st.slider("Threshold Value", 0, 255, 127)
                segmented_img = simple_threshold(img_np, threshold_value)
                
                fig = plot_segmentation_results(img_np, segmented_img, 
                                              f"Simple Threshold (T={threshold_value})")
                st.pyplot(fig)
                
                st.image(segmented_img, caption=f"Thresholded Image (T={threshold_value})", 
                        use_container_width=True, clamp=True)
                
            elif seg_method == "Otsu's Thresholding":
                st.subheader("Otsu's Thresholding")
                segmented_img = otsu_threshold(img_np)
                
                # Calculate the actual threshold value used by Otsu
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                fig = plot_segmentation_results(img_np, segmented_img, 
                                              f"Otsu's Threshold (T={thresh})")
                st.pyplot(fig)
                
                st.image(segmented_img, caption=f"Otsu's Threshold (T={thresh})", 
                        use_container_width=True, clamp=True)
                
                st.markdown("""
                **Otsu's Method** automatically calculates the optimal threshold value 
                by minimizing the intra-class variance.
                """)
                
            elif seg_method == "Adaptive Thresholding":
                st.subheader("Adaptive Thresholding")
                block_size = st.slider("Block Size", 3, 31, 11, step=2)
                C = st.slider("C (constant to subtract)", 0, 10, 2)
                
                segmented_img = adaptive_threshold(img_np, block_size, C)
                
                fig = plot_segmentation_results(img_np, segmented_img, 
                                              f"Adaptive Threshold (Block={block_size}, C={C})")
                st.pyplot(fig)
                
                st.image(segmented_img, 
                        caption=f"Adaptive Threshold (Block={block_size}, C={C})", 
                        use_container_width=True, clamp=True)
                
                st.markdown("""
                **Adaptive Thresholding** calculates different thresholds for different 
                regions of the image, useful for uneven lighting conditions.
                """)
                
            elif seg_method == "K-means Clustering":
                st.subheader("K-means Clustering Segmentation")
                k = st.slider("Number of clusters (k)", 2, 8, 2)
                
                segmented_img, labels, centers = kmeans_segmentation(img_np, k)
                
                fig = plot_segmentation_results(img_np, segmented_img, 
                                              f"K-means Segmentation (k={k})")
                st.pyplot(fig)
                
                st.image(segmented_img, caption=f"K-means Segmentation (k={k})", 
                        use_container_width=True)
                
                # Display cluster centers for color images
                if len(img_np.shape) == 3:
                    st.markdown("**Cluster Centers (RGB values):**")
                    centers_rgb = centers.reshape(-1, 3)
                    for i, center in enumerate(centers_rgb):
                        st.write(f"Cluster {i+1}: R={center[0]}, G={center[1]}, B={center[2]}")
                
                st.markdown("""
                **K-means Clustering** groups pixels into k clusters based on their 
                color similarity, effectively segmenting the image into regions.
                """)   
        with tab_tp7:
            st.header("TP7 - Connected Components Labeling (with Equivalence Table)")
        
        # Create a binary image from the original
            if len(img_np.shape) == 3:
                gray_for_binary = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_for_binary = img_np
        
        # Let user choose threshold method
            threshold_method = st.radio("Threshold Method:", 
                                  ["Simple Threshold", "Otsu's Threshold", "Adaptive Threshold"],
                                  horizontal=True,
                                  key="tp7_threshold")
        
            if threshold_method == "Simple Threshold":
                threshold_value = st.slider("Threshold Value", 0, 255, 127, key="tp7_thresh_val")
                binary_img = simple_threshold(gray_for_binary, threshold_value)
            elif threshold_method == "Otsu's Threshold":
                binary_img = otsu_threshold(gray_for_binary)
            else:  # Adaptive Threshold
                block_size = st.slider("Block Size", 3, 31, 11, step=2, key="tp7_block_size")
                C = st.slider("C (constant to subtract)", 0, 10, 2, key="tp7_c")
                binary_img = adaptive_threshold(gray_for_binary, block_size, C)
        
        # Perform connected components labeling with custom implementation
            labels, colored_labels, num_labels = connected_components_labeling_custom(binary_img)
        
        # Display results
            st.subheader(f"Found {num_labels} connected components")
            fig = plot_components_results_custom(img_np, binary_img, colored_labels, num_labels)
            st.pyplot(fig)
        
        # Show the labeled image
            st.image(colored_labels, caption="Labeled Components", use_container_width=True)
        
            st.markdown("""
        **Connected Components Labeling with Equivalence Table**:
        - First pass: Assign preliminary labels and build equivalence table
        - Second pass: Resolve equivalences and assign final labels
        - Colors represent different connected components
        """)
    
        with tab_tp8:
            st.header("TP8 - Region Properties Calculation")
        
        # First get the binary image and components (same as TP7)
            if len(img_np.shape) == 3:
                gray_for_binary = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_for_binary = img_np
        
        # Let user choose threshold method
            threshold_method = st.radio("Threshold Method:", 
                                  ["Simple Threshold", "Otsu's Threshold", "Adaptive Threshold"],
                                  horizontal=True,
                                  key="tp8_threshold")
        
            if threshold_method == "Simple Threshold":
                threshold_value = st.slider("Threshold Value", 0, 255, 127, key="tp8_thresh_val")
                binary_img = simple_threshold(gray_for_binary, threshold_value)
            elif threshold_method == "Otsu's Threshold":
                binary_img = otsu_threshold(gray_for_binary)
            else:  # Adaptive Threshold
                block_size = st.slider("Block Size", 3, 31, 11, step=2, key="tp8_block_size")
                C = st.slider("C (constant to subtract)", 0, 10, 2, key="tp8_c")
                binary_img = adaptive_threshold(gray_for_binary, block_size, C)
        
        # Perform connected components labeling
            labels, colored_labels, num_labels = connected_components_labeling_custom(binary_img)
        
        # Calculate region properties
            properties = calculate_region_properties(labels, num_labels)
        
        # Display results
            st.subheader(f"Region Properties for {num_labels} components")
        
        # Plot visualization
            fig = plot_region_properties(img_np, properties, colored_labels)
            st.pyplot(fig)
        
        # Show the properties in a table
            st.subheader("Detailed Properties")
        
        # Create a DataFrame for nice display
            props_df = pd.DataFrame([{
            'Label': prop['label'],
            'Area (px)': int(prop['area']),
            'Centroid (x,y)': f"({prop['centroid'][0]:.1f}, {prop['centroid'][1]:.1f})",
            'Bounding Box (x,y,w,h)': prop['bounding_box'],
            'Perimeter (px)': f"{prop['perimeter']:.1f}",
            'Compactness': f"{prop['compactness']:.3f}",
            'Eccentricity': f"{prop['eccentricity']:.3f}"
        } for prop in properties])
        
            st.dataframe(props_df, hide_index=True)
        
            st.markdown("""
        **Region Properties** calculates various metrics for each connected component:
        - Area: Number of pixels in the region
        - Centroid: Center of mass of the region
        - Bounding Box: Rectangle enclosing the region
        - Perimeter: Length of the region's boundary
        - Compactness: How circular the region is (1 = perfect circle)
        - Eccentricity: How elongated the region is (0 = circle, 1 = line segment)
        """)
        with tab_tp9:
                st.header("TP9 - Video Processing")
                st.info("Please upload a video file to use this feature")

        
                   
if __name__ == "__main__":
    main()
