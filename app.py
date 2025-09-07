import streamlit as st
import requests
from PIL import Image
import io

# -------------------------------
# App Title
# -------------------------------
st.title("🌶️ Chili Ripeness Detection")
st.write("Capture or upload an image and let the AI detect the ripeness of chili.")

# -------------------------------
# Image Input Options
# -------------------------------
st.subheader("Upload or Take a Picture")
option = st.radio("Choose input method:", ("📂 Upload Image", "📸 Use Camera"))

img_file = None
if option == "📂 Upload Image":
    img_file = st.file_uploader("Upload a chili photo", type=["jpg", "jpeg", "png"])
elif option == "📸 Use Camera":
    img_file = st.camera_input("Take a picture of your chili")

if img_file is not None:
    # Display image
    st.image(img_file, caption="Selected Image", use_container_width=True)

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
            # Get highest-confidence prediction
            best_pred = max(result["predictions"], key=lambda x: x["confidence"])
            label = best_pred["class"]
            confidence = best_pred["confidence"]

            # ✅ Confidence threshold
            if confidence > 0.8 and "chili" in label.lower():
                st.success(f"✅ Detected Chili: {label} (Confidence: {confidence:.2f})")
            else:
                st.error("❌ Not a chili (low confidence). Try again.")
        
        else:
            st.warning("⚠️ No object detected. Try again with clearer image.")
    else:
        st.error("❌ Error calling Roboflow API. Check API key or project details.")
