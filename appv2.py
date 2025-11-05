# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2  # ✅ Added for BGR to RGB conversion

# -------------------------------
# App Title
# -------------------------------
st.title("🌶️ Chili Ripeness Detection (YOLOv8)")
st.write("Upload or capture an image to detect chili ripeness.")

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
@st.cache_resource
def load_model():
    # 👇 Replace with your trained model path (e.g., 'runs/detect/train/weights/best.pt')
    model_path = "best.pt"
    return YOLO(model_path)

model = load_model()

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
    # Display uploaded or captured image
    image = Image.open(img_file)
    st.image(image, caption="Selected Image", use_container_width=True)

    # -------------------------------
    # Run YOLOv8 Prediction
    # -------------------------------
    results = model.predict(image, conf=0.25, iou=0.4)  # Added conf & iou for cleaner results

    # Show prediction image with bounding boxes
    for result in results:
        annotated_frame = result.plot()  # Draw boxes + labels

        # FIX: Convert BGR → RGB to prevent violet color issue
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated result
        st.image(annotated_frame_rgb, caption="Detection Results", use_container_width=True)

        # -------------------------------
        # Show detected classes and confidence
        # -------------------------------
        st.subheader("Prediction Details")
        if len(result.boxes) == 0:
            st.write("⚠️ No chilis detected.")
        else:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                
