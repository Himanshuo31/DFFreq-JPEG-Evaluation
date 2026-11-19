import io
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

st.set_page_config(
    page_title="DFFreq AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "checkpoints" / "model_epoch_last.pth"

transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@st.cache_resource
def load_model():
    model = resnet50(num_classes=1)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def predict(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    model = load_model()

    with torch.no_grad():
        logit = model(tensor).reshape(-1)[0].item()
        score = torch.sigmoid(torch.tensor(logit)).item()

    label = "AI-generated" if score >= 0.5 else "Real"
    return label, score, logit

def jpeg_image(image, quality):
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=quality
    )
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

st.title("DFFreq: AI-Generated Image Detection")
st.caption("Image forensics • JPEG compression robustness experiment")
st.info(
    "Research prototype: predictions are model outputs, not proof "
    "of an image's authenticity."
)

with st.sidebar:
    st.header("About the project")
    st.write("Model: pretrained DFFreq ResNet-50")
    st.write("Device: CPU")
    st.write("Input preprocessing: center crop 256 × 256")
    st.divider()
    st.write("Upload an image to inspect its prediction.")

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp", "bmp"]
)

if uploaded is not None:
    try:
        original = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded image")
            st.image(original, use_container_width=True)

        with col2:
            st.subheader("Model prediction")
            with st.spinner("Loading model and analysing image..."):
                label, score, logit = predict(original)

            st.metric("Prediction", label)
            st.metric("Model score", f"{score:.6f}")
            st.caption(
                "Score is sigmoid(logit); it is not a calibrated "
                "probability of authenticity."
            )
            st.write(f"Raw logit: `{logit:.6f}`")

        st.divider()
        st.header("JPEG Compression Experiment")
        quality = st.select_slider(
            "JPEG quality factor",
            options=[25, 50, 75, 95],
            value=75
        )

        if st.button("Compare original and compressed image"):
            compressed = jpeg_image(original, quality)
            compressed_label, compressed_score, compressed_logit = (
                predict(compressed)
            )

            left, right = st.columns(2)
            with left:
                st.subheader("Original")
                st.image(original, use_container_width=True)
                st.write(f"Prediction: **{label}**")
                st.write(f"Score: `{score:.6f}`")

            with right:
                st.subheader(f"JPEG quality {quality}")
                st.image(compressed, use_container_width=True)
                st.write(f"Prediction: **{compressed_label}**")
                st.write(f"Score: `{compressed_score:.6f}`")

            if label != compressed_label:
                st.warning("The prediction changed after recompression.")
            else:
                st.success("The prediction remained the same.")

            st.caption(
                "This compares one image. It does not establish "
                "dataset-wide robustness."
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.header("Dataset Experiment Results")

chart_image = ROOT / "accuracy_chart.png"
if chart_image.exists():
    st.subheader("Overall accuracy under JPEG compression")
    st.image(str(chart_image), width=800)

results_path = ROOT / "all_quality_results.csv"
paired_path = ROOT / "paired_results.csv"

if results_path.exists():
    results = pd.read_csv(results_path)
    st.subheader("Overall accuracy")
    if "Accuracy" in results.columns and "Quality" in results.columns:
        chart_data = results[["Quality", "Accuracy"]].copy()
        chart_data["Quality"] = chart_data["Quality"].astype(str)
        chart_data["Accuracy (%)"] = chart_data["Accuracy"].astype(float) * 100
        st.bar_chart(chart_data, x="Quality", y="Accuracy (%)")
        st.caption("Overall accuracy (%) by image condition")
    st.dataframe(results, use_container_width=True)
else:
    st.warning("all_quality_results.csv was not found.")

if paired_path.exists():
    st.subheader("Paired predictions")
    st.dataframe(pd.read_csv(paired_path), use_container_width=True)
else:
    st.warning("paired_results.csv was not found.")

st.caption(
    "Important limitation: the original real images are JPG and "
    "the original fake images are PNG. This format difference may "
    "confound the compression experiment."
)



