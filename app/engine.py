import torch
from diffusers import StableDiffusionPipeline
import clip
from ultralytics import SAM
from PIL import Image
import numpy as np
from app.config import Config
from app.utils import mask_to_polygon, crop_image_by_mask , image_to_base64
from app.cifar_labels import CIFAR_100_CLASSES


class AIEngine:
    def __init__(self):
        print("⏳ Loading AI Models... (This may take time)")
        
        # 1. Load Stable Diffusion
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            Config.SD_MODEL_ID, 
            torch_dtype=Config.DTYPE
        ).to(Config.DEVICE)

        # Disable safety checker for assignment purposes (avoids black images)
        self.sd_pipe.safety_checker = None 

        # 2. Load CLIP
        self.clip_model, self.clip_preprocess = clip.load(
            Config.CLIP_MODEL_ID, 
            device=Config.DEVICE
        )

        # 3. Load SAM 2
        self.sam_model = SAM(Config.SAM_MODEL_ID)

        print("🧠 Pre-calculating CIFAR-100 embeddings...")
        self.cifar_labels = CIFAR_100_CLASSES

        text_prompts = [f"a photo of a {label}" for label in self.cifar_labels]
        
        text_inputs = clip.tokenize(text_prompts).to(Config.DEVICE)
        
        with torch.no_grad():
            # Calculate features ONCE
            text_features = self.clip_model.encode_text(text_inputs)
            # Normalize them ONCE
            self.cifar_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        print("✅ Models & Embeddings Ready.")

    def generate_image(self, prompt: str):
        """Generates an image from text."""
        steps = 15 if Config.DEVICE == "cuda" else 15 # Optimization for CPU
        image = self.sd_pipe(prompt, num_inference_steps=steps).images[0]
        return image

    def classify_cifar100(self, image: Image.Image):
        """
        Compares image against pre-cached CIFAR-100 vectors.
        Returns Top 5 classes with scores.
        """
        # 1. Process Image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            # 2. Get Image Features
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 3. Compare with Pre-computed CIFAR features (Matrix Multiplication)
            # Shape: (1, 512) @ (100, 512).T = (1, 100)
            similarity = (100.0 * image_features @ self.cifar_features.T).softmax(dim=-1)
            
            # 4. Get Top 5 Results
            # values: scores, indices: which class index (0-99)
            values, indices = similarity[0].topk(5)

            top_5 = []

            for value, index in zip(values, indices):
                top_5.append({
                    "class": self.cifar_labels[index],
                    "confidence": f"{value.item() * 100:.2f}%"
                })
                
            return top_5

    def analyze_image(self, image: Image.Image, concepts: list,softmax=True):
        """
        Uses CLIP to score the image against a list of text concepts.
        Returns a dictionary of {concept: confidence_score}.
        """
        image_input = self.clip_preprocess(image).unsqueeze(0).to(Config.DEVICE)
        text_inputs = clip.tokenize(concepts).to(Config.DEVICE)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        if softmax:
            # OLD LOGIC: Relative (Good for comparing "cat" vs "dog")
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores = similarity.cpu().numpy()[0]
            # Returns e.g. 99.0, 1.0 (Sums to 100)
        else:
            # NEW LOGIC: Absolute (Good for checking "is this a cat?")
            similarity = (image_features @ text_features.T)
            scores = similarity.cpu().numpy()[0]
            # Returns raw cosine e.g. 0.26, 0.15 (Does not sum to 100)
            # We multiply by 100 just to make it readable as a percentage
            scores = (scores - 0.22) / (0.33 - 0.22) * 100
            scores = scores.clip(0, 100)

        return {concept: float(score) for concept, score in zip(concepts, scores)}

    def segment_image(self, image: Image.Image):
        """
        Uses SAM2 to find regions. Returns masks and polygons.
        """
        # Run inference
        results = self.sam_model(image, device=Config.DEVICE, verbose=False)
        result = results[0]
        
        segments = []
        if result.masks:
            masks_data = result.masks.data.cpu().numpy() # Get raw mask arrays
             
            for i, mask in enumerate(masks_data):
                # Advanced Feature: Polygons
                polygons = mask_to_polygon(mask)
                segments.append({
                    "id": i,
                    "polygon": polygons,
                    "mask_shape": mask.shape
                })
                
        return segments, result # Return result object for advanced probing

    def probe_regions(self, image: Image.Image, concepts: list):
        """
        Advanced Feature: Concept Probing on Segments.
        Segments the image, then checks each segment against the concepts.
        """
        segments, sam_result = self.segment_image(image)
        regional_analysis = []

        if sam_result.masks:
            masks_data = sam_result.masks.data.cpu().numpy()
            
            # Limit to top 5 largest regions to save time
            for i, mask in enumerate(masks_data[:5]):
                # Crop the region
                crop = crop_image_by_mask(image, mask)
                # Analyze the crop with CLIP
                scores = self.analyze_image(crop, concepts)
                
                regional_analysis.append({
                    "region_id": i,
                    "scores": scores
                })
        
        return regional_analysis
    
    def segment_image(self, image: Image.Image, conf=0.95):
        """
        Uses SAM2 to find regions. 
        Filters out any regions with confidence score < conf.
        """
        # Pass the 'conf' argument directly to the model
        results = self.sam_model(image, device=Config.DEVICE, verbose=False, conf=conf)
        result = results[0]
        
        segments = []
        if result.masks:
            masks_data = result.masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks_data):
                polygons = mask_to_polygon(mask)
                segments.append({
                    "id": i,
                    "polygon": polygons,
                    # We will create a simple label for the mask
                    "mask_label": f"region_{i}" 
                })
                
        return segments, result
    
    def process_detailed_segmentation(self, image: Image.Image, conf=0.95):
        """
        1. Segments the image.
        2. Crops each segment.
        3. Classifies each crop individually.
        4. Returns the detailed JSON structure.
        """
        # Run SAM2 with high confidence threshold
        results = self.sam_model(image, device=Config.DEVICE, verbose=False, conf=conf)
        result = results[0]
        
        detailed_regions = []
        
        if result.masks:
            masks_data = result.masks.data.cpu().numpy()
            conf_scores = result.boxes.conf.cpu().numpy() if result.boxes else []
            
            for i, mask in enumerate(masks_data):
                # A. Crop the specific region (Black out background)
                crop = crop_image_by_mask(image, mask)
                
                # B. Analyze ONLY this region
                region_analysis = self.classify_cifar100(crop)
                
                # C. Get Polygon
                polygon = mask_to_polygon(mask)
                
                # D. Convert Crop to Base64
                crop_b64 = image_to_base64(crop)

                mask_conf = float(conf_scores[i]) if i < len(conf_scores) else 0.0
                
                detailed_regions.append({
                    "region_id": i,
                    "sam_confidence": f"{mask_conf:.2f}",
                    "clip_analysis": region_analysis, # Specific to this crop
                    "polygon": polygon,
                    "image": crop_b64
                })
                
        return detailed_regions
    
