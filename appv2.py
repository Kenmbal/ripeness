import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
from collections import Counter

st.title("🌶️ Chili Ripeness Detection (YOLOv8)")
st.write("Upload or capture an image to detect chili ripeness.")

@st.cache_resource
def load_model():
    model_path = "best.pt"
    return YOLO(model_path)

model = load_model()

st.subheader("Upload or Take a Picture")
option = st.radio("Choose input method:", ("📂 Upload Image", "📸 Use Camera"))

img_file = None
if option == "📂 Upload Image":
    img_file = st.file_uploader("Upload a chili photo", type=["jpg", "jpeg", "png"])
elif option == "📸 Use Camera":
    img_file = st.camera_input("Take a picture of your chili")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Selected Image", use_container_width=True)

    results = model.predict(image, conf=0.25, iou=0.4)

    for result in results:
        annotated_frame = result.plot(conf=False)
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_frame_rgb, caption="Detection Results", use_container_width=True)

        st.subheader("Summary")

        if len(result.boxes) == 0:
            st.write("⚠️ No chilis detected.")
        else:
            # Collect all detected labels
            labels = [model.names[int(box.cls[0])] for box in result.boxes]

            # Count each label
            counts = Counter(labels)

            # Display summary neatly
            total = sum(counts.values())
            for label, count in counts.items():
                st.write(f"✅ **{label} – {count}**")

            st.write(f"📊 **Total Detected Chilis: {total}**")
