import gradio as gr
import requests
import base64
import json
from io import BytesIO
from PIL import Image

# Configuration
API_URL = "http://127.0.0.1:8000"

# --- Helper Functions ---
def decode_image(b64_string):
    """Converts Base64 string back to PIL Image."""
    if not b64_string: return None
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data))

# --- 1. GENERATE ---
def action_generate(prompt):
    """Calls /generate"""
    try:
        response = requests.post(f"{API_URL}/generate", json={"prompt": prompt})
        if response.status_code != 200:
            return None, f"❌ Error: {response.text}"
        
        data = response.json()
        main_image = decode_image(data["generated_image"])
        return main_image, "✅ Image Generated! Proceed to next steps."
    except Exception as e:
        return None, f"❌ Connection Error: {str(e)}"

# --- 2. ANALYZE (CIFAR-100) ---
def action_analyze():
    """Calls /analyze to get Top 5 Classes"""
    try:
        response = requests.post(f"{API_URL}/analyze")
        if response.status_code != 200:
            return f"❌ Error: {response.text}"
            
        data = response.json()
        scores = data.get("clip_scores", [])
        
        # Format as Markdown List
        output = "### 📊 Top 5 Predicted Classes\n"
        for item in scores:
            output += f"- **{item['class']}**: {item['confidence']}\n"
            
        return output
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

# --- 3. PROBE (Similarity) ---
def action_probe(text_prompt):
    """Calls /probe to compare text against image & regions"""
    try:
        response = requests.post(f"{API_URL}/probe", json={"prompt": text_prompt})
        if response.status_code != 200:
            return f"❌ Error: {response.text}"
        
        data = response.json()
        
        md = f"### 🔎 Probe Results for: '{data['probe_text']}'\n"
        md += f"**🌍 Global Image Match:** `{data['global_match_score']}`\n\n"
        md += "| ID | Class | Match Score |\n"
        md += "|----|-------|-------------|\n"
        
        for region in data["regional_matches"]:
            md += f"| {region['region_id']} | {region['class_label']} | **{region['similarity_to_prompt']}** |\n"
            
        return md
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

# --- 4. SEGMENT (JSON Data) ---
def action_segment():
    """Calls /segment to get Polygons & Data"""
    try:
        response = requests.post(f"{API_URL}/segment")
        if response.status_code != 200:
            return {"error": response.text}
            
        data = response.json()
        # Return the full JSON (Gradio displays this nicely)
        return data["segmented_regions"]
    except Exception as e:
        return {"error": str(e)}

# --- 5. VISUALIZE (Images) ---
def action_visualize():
    """Calls /visualize to get Overlay & Spotlights"""
    try:
        response = requests.post(f"{API_URL}/visualize")
        if response.status_code != 200:
            return None, [], f"❌ Error: {response.text}"
            
        data = response.json()
        overlay = decode_image(data["master_overlay"])
        
        gallery = []
        for item in data["individual_spotlights"]:
            img = decode_image(item["spotlight_image"])
            label = f"ID: {item['region_id']} ({item['top_class']})"
            gallery.append((img, label))
            
        return overlay, gallery, "✅ Visualization Loaded"
    except Exception as e:
        return None, [], f"❌ Connection Error: {str(e)}"

# --- UI LAYOUT ---
with gr.Blocks(title="AI Pipeline Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Full AI Pipeline Control Center")
    
    # SECTION 1: GENERATE
    with gr.Group():
        gr.Markdown("### 1️⃣ Generate Image")
        with gr.Row():
            txt_prompt = gr.Textbox(label="Prompt", value="a futuristic cyberpunk city street with neon lights")
            btn_gen = gr.Button("Generate", variant="primary")
        
        out_image = gr.Image(label="Generated Image", type="pil", height=400)
        out_status = gr.Markdown("")
        
        btn_gen.click(action_generate, inputs=txt_prompt, outputs=[out_image, out_status])

    gr.Markdown("---") # Divider

    # SECTION 2: ANALYZE
    with gr.Group():
        gr.Markdown("### 2️⃣ Analyze (CIFAR-100 Classification)")
        btn_analyze = gr.Button("Analyze Image")
        out_analysis = gr.Markdown("Results will appear here...")
        
        btn_analyze.click(action_analyze, inputs=None, outputs=out_analysis)

    gr.Markdown("---")

    # SECTION 3: PROBE
    with gr.Group():
        gr.Markdown("### 3️⃣ Probe (Concept Similarity)")
        with gr.Row():
            txt_probe = gr.Textbox(label="Concept to check", placeholder="e.g. red, metal, dark")
            btn_probe = gr.Button("Run Probe")
        out_probe = gr.Markdown("Probe results...")
        
        btn_probe.click(action_probe, inputs=txt_probe, outputs=out_probe)

    gr.Markdown("---")

    # SECTION 4: SEGMENT
    with gr.Group():
        gr.Markdown("### 4️⃣ Segment Data (Polygons & JSON)")
        btn_segment = gr.Button("Get Segmentation Data")
        out_json = gr.JSON(label="Raw Segmentation Output")
        
        btn_segment.click(action_segment, inputs=None, outputs=out_json)

    gr.Markdown("---")

    # SECTION 5: VISUALIZE
    with gr.Group():
        gr.Markdown("### 5️⃣ Visualize (Overlay & Spotlights)")
        btn_vis = gr.Button("Visualize Regions")
        
        with gr.Row():
            out_overlay = gr.Image(label="Master Overlay", type="pil")
            out_gallery = gr.Gallery(label="Individual Segments", columns=4)
        
        btn_vis.click(action_visualize, inputs=None, outputs=[out_overlay, out_gallery, out_status])

if __name__ == "__main__":
    demo.launch(server_port=7860)