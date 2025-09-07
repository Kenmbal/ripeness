import streamlit as st
import requests
from PIL import Image
import io

# -------------------------------
# App Title
# -------------------------------
st.title("🌶️ Chili Ripeness Detection")
st.write("Capture an image using your camera and let the AI detect the ripeness of chili.")

# -------------------------------
# Camera Input
# -------------------------------
img_file = st.camera_input("Take a picture of your chili")

if img_file is not None:
    # Display image
    st.image(img_file, caption="Captured Image", use_container_width=True)

    # Convert image for sending to Roboflow API
    image = Image.open(img_file)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    # -------------------------------
    # Roboflow API Prediction
    # -------------------------------
    API_KEY = "4UetNZJemq4mrCoAdS5C"  # 🔑 replace with your Roboflow API key
    PROJECT = "crop-harvesting-prediction-wki2v"  # your project name
    VERSION = 1

    upload_url = f"https://detect.roboflow.com/{PROJECT}/{VERSION}?api_key={API_KEY}"

    response = requests.post(upload_url, files={"file": buffered})
    
    if response.status_code == 200:
        result = response.json()
        
        # Parse results
        st.subheader("Prediction Results")

        if "predictions" in result and len(result["predictions"]) > 0:
            chili_detected = False

            for pred in result["predictions"]:
                label = pred["class"]
                confidence = pred["confidence"]

                # ✅ Check if object is chili
                if "chili" in label.lower():
                    chili_detected = True
                    st.success(f"✅ Detected Chili: {label} (Confidence: {confidence:.2f})")

            if not chili_detected:
                st.error("❌ Error: Object detected is not a chili. Please try again.")
        
        else:
            st.warning("⚠️ No object detected. Try again with clearer image.")
    else:
        st.error("❌ Error calling Roboflow API. Check API key or project details.")
