import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 string for JSON response."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    """Converts a Base64 string back to a PIL Image."""
    return Image.open(BytesIO(base64.b64decode(base64_str)))

def mask_to_polygon(binary_mask: np.ndarray):
    """
    Advanced Feature: Converts a binary mask (black/white) into 
    polygon coordinates [[x,y], [x,y]] for visualization.
    """
    # Ensure mask is uint8
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    # Find contours (boundaries)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Simplify contour to reduce data size (optional but good for API)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Flatten and convert to list of [x, y]
        points = approx.reshape(-1, 2).tolist()
        if len(points) > 2: # A polygon needs at least 3 points
            polygons.append(points)
            
    return polygons

def crop_image_by_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Advanced Feature: Crops specific region for Concept Probing."""
    image_np = np.array(image)
    # Create an RGBA image with transparency
    rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = (mask * 255).astype(np.uint8) # Set alpha channel based on mask
    
    # Find bounding box
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0: return image # Fallback
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Crop
    cropped = rgba[y_min:y_max+1, x_min:x_max+1]
    return Image.fromarray(cropped)
def draw_segmentation_overlay(image: Image.Image, regions: list) -> Image.Image:
    """
    Draws green polygon outlines on top of the image for visualization.
    FIXED: Handles regions containing multiple disconnected shapes.
    """
    image_np = np.array(image)
    overlay = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for region in regions:
        # 'polygons' is a LIST of contours (e.g. [[x,y...], [x,y...]])
        polygons = region["polygon"] 
        if not polygons: continue
        
        # Iterate over each distinct shape in this region
        for poly in polygons:
            pts = np.array(poly, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw the Polygon
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Add Label ID text (put it at the start of the largest shape)
        if len(polygons) > 0:
            # Find largest shape to place text
            largest_poly = max(polygons, key=len)
            start_point = tuple(np.array(largest_poly[0], int))
            cv2.putText(overlay, f"ID:{region['region_id']}", start_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def create_spotlight_image(image: Image.Image, polygons: list) -> Image.Image:
    """
    Creates an image where the whole area is dimmed 
    EXCEPT for the region defined by the polygon.
    FIXED: Handles regions with multiple shapes.
    """
    image_rgba = image.convert("RGBA")
    image_np = np.array(image_rgba)
    height, width = image_np.shape[:2]

    # Start with black mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if polygons:
        # Prepare list of arrays for fillPoly
        # cv2.fillPoly expects a list of numpy arrays [array1, array2]
        cv_polys = []
        for poly in polygons:
            cv_polys.append(np.array(poly, np.int32))
            
        # Fill all shapes in this region with white
        cv2.fillPoly(mask, cv_polys, color=255)

    # Create Dimmed Background
    dimmed_bg = image_np.copy()
    dimmed_bg[:, :, 3] = (dimmed_bg[:, :, 3] * 0.3).astype(np.uint8)

    # Combine
    final_np = np.where(mask[..., None] == 255, image_np, dimmed_bg)

    return Image.fromarray(final_np)