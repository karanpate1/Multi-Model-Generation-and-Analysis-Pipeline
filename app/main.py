from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.engine import AIEngine
from app.utils import image_to_base64, base64_to_image, draw_segmentation_overlay, create_spotlight_image
from typing import List, Optional
import uuid
from PIL import Image
import logging

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),          # Output to Console
        logging.FileHandler("app.log")    # Output to File (Optional)
    ]
)
logger = logging.getLogger(__name__)

# Initialize App and Engine
app = FastAPI(title="AI Multi-Model Pipeline")
engine = AIEngine()

# --- Data Models (Pydantic) ---
class GenerateRequest(BaseModel):
    prompt: str

class SingleProbeRequest(BaseModel):
    prompt: str

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "device": str(engine.sd_pipe.device)}
last_generated_image = None
last_segmented_regions = None
@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """
    1. Generate Image (Stable Diffusion)
    2. Analyze Image (CLIP - verifies against prompt)
    3. Segment Image (SAM2 - basic masks)
    """
    global last_generated_image,last_segmented_regions 
    last_segmented_regions = None

    try:
        request_id = str(uuid.uuid4())
        
        # 1. Generate
        image = engine.generate_image(request.prompt)
        last_generated_image = image
        img_b64 = image_to_base64(image)
        
        # 2. Classify (CIFAR-100 Check)
        # This is now super fast because text embeddings are cached!
        cifar_analysis = engine.classify_cifar100(image)
        
        return {
            "request_id": request_id,
            "generated_image": img_b64,
            "clip_analysis": cifar_analysis,
        }
    except Exception as e:
        logger.error(f"❌ Error in /generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_endpoint(): # No input arguments needed now!
    """
    Analyzes the LAST generated image.
    1. Checks if image exists.
    2. CIFAR-100 Top 5 Classification.
    3. SAM2 Segmentation (Confidence > 0.95).
    """
    global last_generated_image, last_segmented_regions
    
    # 1. Check if image exists
    if last_generated_image is None:
        raise HTTPException(
            status_code=400, 
            detail="No image found! Please generate image first."
        )

    try:
        request_id = str(uuid.uuid4())
        image = last_generated_image # Get from memory

        # 2. CLIP Analysis (CIFAR-100 Top 5)
        cifar_analysis = engine.classify_cifar100(image)
        
        # 3. Segmentation (Strict Confidence > 0.9)
        # We pass 0.90 to filter out weak segments
        segments, _ = engine.segment_image(image, conf=0.95)
        
        # Format the output exactly as requested
        return {
            "request_id": request_id,
            "clip_scores": cifar_analysis, # Top 5 classes
            "segmentation": {
                "masks": [s["mask_label"] for s in segments], # List of IDs
                "polygons": [s["polygon"] for s in segments]  # List of coordinates
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in /analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/probe")
async def probe_endpoint(request: SingleProbeRequest):
    """
    Checks similarity of ONE text prompt against:
    1. The whole image.
    2. Each specific segmented region.
    """
    global last_generated_image, last_segmented_regions
    
    # 1. Validation
    if last_generated_image is None:
        raise HTTPException(status_code=400, detail="No image found! Call /generate first.")

    try:
        # 2. Ensure Segmentation Data Exists
        if last_segmented_regions is None:
            logger.info(" Computing Segmentation for probe...")
            last_segmented_regions = engine.process_detailed_segmentation(last_generated_image, conf=0.95 , )

        # 3. Analyze Whole Image (Global Score)
        # analyze_image returns dict like {"text": 0.95}, we just want the float value
        global_result = engine.analyze_image(last_generated_image, [request.prompt],softmax=False)
        global_score = global_result[request.prompt]

        # 4. Analyze Each Region (Local Scores)
        region_scores = []
        
        for region in last_segmented_regions:
            # We must convert the Base64 crop back to an Image to run CLIP on it
            crop_image = base64_to_image(region["image"])
            
            # Run CLIP on this specific crop
            local_result = engine.analyze_image(crop_image, [request.prompt],softmax=False)
            score = local_result[request.prompt]
            
            region_scores.append({
                "region_id": region["region_id"],
                "class_label": region["clip_analysis"][0]["class"], # Helpful context
                "similarity_to_prompt": f"{score:.2f}%"
            })

        return {
            "probe_text": request.prompt,
            "global_match_score": f"{global_score:.2f}%",
            "regional_matches": region_scores
        }

    except Exception as e:
        logger.error(f"❌ Error in /probe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/segment")
async def segment_endpoint():
    """
    Detailed Segmentation:
    1. Checks for last generated image.
    2. Segments it.
    3. Crops & Classifies EACH segment separately.
    """
    global last_generated_image, last_segmented_regions
    
    if last_generated_image is None:
        raise HTTPException(
            status_code=400, 
            detail="No image found! Please call /generate first."
        )

    try:
        request_id = str(uuid.uuid4())
        
        # Run the detailed analysis (Threshold 0.85 for decent object detection)
        # You can adjust 'conf' higher if you want fewer, strictly clearer objects
        if last_segmented_regions is None:
            logger.info(" Computing Segmentation for Visualization... for request %s", request_id)
            # Using a slightly higher confidence for visualization so it looks cleaner
            last_segmented_regions = engine.process_detailed_segmentation(last_generated_image, conf=0.95)
        clean_regions = []
        for region in last_segmented_regions:
            # Create a new dictionary excluding the 'image' key
            clean_item = {k: v for k, v in region.items() if k != "image"}
            clean_regions.append(clean_item)

        return {
            "request_id": request_id,
            "segmented_regions": clean_regions
        }
    except Exception as e:
        logger.error(f"❌ Error in /segment endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def visualize_endpoint():
    """
    Returns:
    1. 'master_overlay': Image with outlines of ALL regions.
    2. 'individual_spotlights': A list of images, one for each region, 
       where background is dim and only that region is highlighted.
    """
    global last_generated_image, last_segmented_regions
    
    if last_generated_image is None:
        raise HTTPException(status_code=400, detail="No image! Call /generate first.")

    try:
        # A. Ensure Cache Exists (same as before)
        if last_segmented_regions is None:
            logger.info(" Computing Segmentation for Visualization... for request ")
            # Using a slightly higher confidence for visualization so it looks cleaner
            last_segmented_regions = engine.process_detailed_segmentation(last_generated_image, conf=0.95)
        
        # --- PART 1: Master Overlay Image (All outlines) ---
        overlay_image = draw_segmentation_overlay(last_generated_image, last_segmented_regions)
        overlay_b64 = image_to_base64(overlay_image)
        
        # --- PART 2: Individual Spotlight Images ---
        spotlight_list = []
        logger.info(f"🎨 Generating {len(last_segmented_regions)} spotlight images...")
        
        for region in last_segmented_regions:
            # Generate spotlight using the new utility function
            spotlight_img = create_spotlight_image(last_generated_image, region["polygon"])
            
            spotlight_list.append({
                "region_id": region["region_id"],
                # We can include the class name for context if available
                "top_class": region["clip_analysis"][0]["class"], 
                "polygon": region["polygon"],
                "spotlight_image": image_to_base64(spotlight_img)
            })
            
        return {
            "status": "success",
            "total_regions": len(spotlight_list),
            "master_overlay": overlay_b64,        # The single main image
            "individual_spotlights": spotlight_list # The list of individual images
        }
        
    except Exception as e:
        logger.error(f"❌ Error in /visualize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))