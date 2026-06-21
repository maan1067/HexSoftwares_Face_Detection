import streamlit as st
import cv2
import numpy as np
import av
import time
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="AI Face Detection System",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* ── Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #0d1f3c 0%, #060b18 55%, #0a0e1a 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer { visibility: hidden; }

/* ── Title ── */
@keyframes pulse-glow {
    0%   { text-shadow: 0 0 8px #3B82F6, 0 0 20px #1D4ED8; }
    50%  { text-shadow: 0 0 20px #60A5FA, 0 0 50px #3B82F6; }
    100% { text-shadow: 0 0 8px #3B82F6, 0 0 20px #1D4ED8; }
}

h1 {
    text-align: center;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E0EFFF !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    animation: pulse-glow 3s ease-in-out infinite;
    margin-bottom: 0.2rem !important;
}

/* ── Subtitle ── */
.subtitle {
    text-align: center;
    color: #7FA8D4 !important;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* ── General text ── */
p, label, span, div, h2, h3, h4, h5, h6 {
    color: #CBD5E1 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Info banner ── */
[data-testid="stAlert"] {
    background: rgba(59,130,246,0.12) !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    border-radius: 12px !important;
    color: #BAD7FF !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #94A3B8 !important;
    border-radius: 10px 10px 0 0 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom: 3px solid #3B82F6 !important;
    background: rgba(59,130,246,0.08) !important;
}

/* ── Upload Box ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(96,165,250,0.5) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    background: linear-gradient(
        135deg,
        rgba(14,30,60,0.9),
        rgba(20,40,80,0.6)
    ) !important;
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #60A5FA !important;
}

/* Label "Choose an image" */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] span {
    color: #93C5FD !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Hint text "200MB per file • JPG, PNG" */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #64748B !important;
}

/* Upload button inside the box */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.8rem !important;
    box-shadow: 0 3px 12px rgba(59,130,246,0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    position: relative !important;
    overflow: hidden !important;
}

/* إخفاء كل النصوص الداخلية المتكررة */
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploaderDropzone"] button p {
    visibility: hidden !important;
    font-size: 0 !important;
    display: block !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* إظهار نص واحد نظيف */
[data-testid="stFileUploader"] button::after,
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Browse File" !important;
    visibility: visible !important;
    color: #EFF6FF !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    position: absolute !important;
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
    white-space: nowrap !important;
}

[data-testid="stFileUploader"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 18px rgba(59,130,246,0.55) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
    border-radius: 999px !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f2042, #172a50) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    border: 1px solid rgba(96,165,250,0.25) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

[data-testid="stMetricValue"] {
    color: #34D399 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #93C5FD !important;
    font-weight: 600 !important;
}

/* ── Success / Error Messages ── */
[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(52,211,153,0.35) !important;
    border-radius: 12px !important;
    color: #6EE7B7 !important;
}

/* ── Webcam video ── */
video {
    border-radius: 18px !important;
    border: 2px solid rgba(96,165,250,0.6) !important;
    box-shadow: 0 0 30px rgba(59,130,246,0.3) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    width: 100% !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #059669, #10B981) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    border: none !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.35) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(16,185,129,0.5) !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(96,165,250,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #4B6A8A !important;
    font-size: 0.85rem;
    margin-top: 1.5rem;
    letter-spacing: 0.3px;
}

.footer span {
    color: #60A5FA !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #93C5FD !important;
}

</style>
""", unsafe_allow_html=True)

# ── Header (single, clean) ──
st.markdown("<h1>🤖 AI Face Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Upload an image or use your webcam to detect faces in real time</p>",
    unsafe_allow_html=True
)

st.info("📷 Powered by OpenCV Haar Cascade — fast, accurate, runs locally.")

# ── Load classifier ──
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Live Camera"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Image Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        with st.spinner("🤖 Scanning for faces…"):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                progress.progress(i + 1)
            progress.empty()

        image = Image.open(uploaded_file)
        img   = np.array(image)
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            # Bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (96, 165, 250), 3)
            # Label background
            cv2.rectangle(img, (x, y - 28), (x + 70, y), (96, 165, 250), -1)
            cv2.putText(
                img, "Face",
                (x + 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (10, 20, 40), 2
            )

        st.image(img, caption=f"Detected {len(faces)} face(s)", use_container_width=True)
        st.success(f"✅ Found **{len(faces)}** face(s) in the image.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("👤 Faces Found", len(faces))
        with col2:
            st.metric("🤖 AI Status", "Active")

        success, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        st.download_button(
            label="📥 Download Result Image",
            data=buffer.tobytes(),
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Live Camera
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:

    st.write("Allow camera access — detection starts automatically.")

    class FaceDetector(VideoProcessorBase):

        def recv(self, frame):
            img  = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (96, 165, 250), 3)
                cv2.rectangle(img, (x, y - 28), (x + 70, y), (96, 165, 250), -1)
                cv2.putText(
                    img, "Face",
                    (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (10, 20, 40), 2
                )

            cv2.putText(
                img, f"Faces: {len(faces)}",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (96, 165, 250), 2
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="face-detection",
        video_processor_factory=FaceDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ── Footer ──
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        🚀 AI Face Detection System &nbsp;·&nbsp;
        <span>OpenCV</span> &nbsp;·&nbsp;
        <span>Streamlit</span> &nbsp;·&nbsp;
        <span>Haar Cascade</span>
    </div>
    """,
    unsafe_allow_html=True
)