import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import time
import requests
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visio — Image Classifier",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inject custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&family=Outfit:wght@200;300;400&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080b12 !important;
    color: #e8e0d0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 300;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(180,148,80,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(40,60,120,0.12) 0%, transparent 55%),
        #080b12 !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    max-width: 780px !important;
    padding: 3rem 2rem 5rem !important;
}

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Cormorant Garamond', serif !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #b49450;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: block;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 4.2rem;
    font-weight: 300;
    line-height: 1.05;
    color: #f0e8d8;
    letter-spacing: -0.01em;
    margin-bottom: 0.4rem;
}
.hero-title em {
    font-style: italic;
    color: #c8a85a;
}
.hero-subtitle {
    font-size: 0.9rem;
    font-weight: 200;
    color: #8a8070;
    letter-spacing: 0.06em;
    margin-top: 0.5rem;
}
.hero-line {
    width: 48px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #b49450, transparent);
    margin: 1.8rem auto 0;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.018) !important;
    border: 1px solid rgba(180,148,80,0.2) !important;
    border-radius: 16px !important;
    padding: 2.5rem !important;
    transition: border-color 0.3s, background 0.3s !important;
    backdrop-filter: blur(10px);
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(180,148,80,0.45) !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    color: #8a8070 !important;
    text-transform: uppercase !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.4rem !important;
    color: #c8b89a !important;
    font-weight: 300 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #6a6058 !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: 1px solid rgba(180,148,80,0.35) !important;
    color: #b49450 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.25s !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(180,148,80,0.08) !important;
    border-color: #b49450 !important;
}

/* ── Image preview card ── */
.img-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    overflow: hidden;
    margin: 1.8rem 0 0;
    position: relative;
}
.img-card-label {
    position: absolute;
    top: 14px; left: 14px;
    background: rgba(8,11,18,0.75);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(180,148,80,0.3);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: #b49450;
    text-transform: uppercase;
    z-index: 10;
}
[data-testid="stImage"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}
[data-testid="stImage"] img {
    border-radius: 16px !important;
}

/* ── Classify button ── */
[data-testid="stBaseButton-primary"],
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, rgba(180,148,80,0.12), rgba(180,148,80,0.06)) !important;
    border: 1px solid rgba(180,148,80,0.5) !important;
    color: #c8a85a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    margin-top: 1.4rem !important;
    transition: all 0.3s !important;
    backdrop-filter: blur(6px) !important;
}
[data-testid="stBaseButton-primary"]:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(180,148,80,0.22), rgba(180,148,80,0.12)) !important;
    border-color: #c8a85a !important;
    box-shadow: 0 0 28px rgba(180,148,80,0.18) !important;
    transform: translateY(-1px) !important;
}

/* ── Results panel ── */
.results-panel {
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    margin-top: 2rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.results-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,148,80,0.4), transparent);
}
.result-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #6a6058;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.top-prediction {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    margin-bottom: 1.8rem;
}
.top-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 400;
    color: #f0e8d8;
    line-height: 1;
}
.top-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #b49450;
    font-weight: 300;
}

/* ── Bar chart rows ── */
.pred-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.pred-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: #c8b89a;
    width: 130px;
    flex-shrink: 0;
    letter-spacing: 0.03em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pred-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #b49450, #e8c87a);
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.pred-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #7a7060;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Meta strip ── */
.meta-strip {
    display: flex;
    gap: 2rem;
    margin-top: 2rem;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    flex-wrap: wrap;
}
.meta-item { display: flex; flex-direction: column; gap: 0.2rem; }
.meta-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    color: #5a5448;
    text-transform: uppercase;
}
.meta-val {
    font-size: 0.82rem;
    color: #a09080;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Divider ── */
.gold-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,148,80,0.25), transparent);
    margin: 2rem 0;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: #b49450 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(180,148,80,0.2); border-radius: 2px; }

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ── FastAPI endpoint config ────────────────────────────────────────────────────
FASTAPI_URL = "http://13.51.48.170:8000/predict"  # 🔧 Change host/port if needed


def classify_image(image: Image.Image, filename: str = "image.jpg"):
    """
    Sends the image to the FastAPI /predict endpoint and returns
    a dict with 'class_name', 'prediction' (index), and optional 'confidence'.
    API response format: {"prediction": 3, "class_name": "Tomato___Bacterial_spot"}
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    try:
        response = requests.post(
            FASTAPI_URL,
            files={"file": (filename, buf, "image/jpeg")},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Primary format: {"prediction": 3, "class_name": "Tomato___Bacterial_spot"}
        if "class_name" in data:
            raw = data["class_name"]
            # Clean up underscores/triple-underscores e.g. "Tomato___Bacterial_spot"
            # → "Tomato — Bacterial Spot"
            parts = raw.replace("___", "|||").replace("_", " ").split("|||")
            if len(parts) == 2:
                display = f"{parts[0].strip().title()} — {parts[1].strip().title()}"
            else:
                display = raw.replace("_", " ").title()
            return {
                "class_name": display,
                "raw_class": raw,
                "prediction_index": data.get("prediction"),
                "confidence": data.get("confidence"),   # None if not returned
            }

        # Fallback: just a prediction string
        if "prediction" in data:
            val = str(data["prediction"]).replace("_", " ").title()
            return {"class_name": val, "raw_class": val, "prediction_index": None, "confidence": None}

        return {"class_name": str(data), "raw_class": str(data), "prediction_index": None, "confidence": None}

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the FastAPI server. Is it running?")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out. The model may still be loading.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ Server error: {e.response.status_code} — {e.response.text}")
        st.stop()


def get_image_meta(image: Image.Image, filename: str) -> dict:
    return {
        "dimensions": f"{image.width} × {image.height}",
        "mode":       image.mode,
        "filename":   filename,
    }


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-badge"></span>
    <div class="hero-title"><em>Dezznut Ai</em></div>
    <div class="hero-subtitle">Plant Disease Classification Engine</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── File uploader ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

# ── Main flow ──────────────────────────────────────────────────────────────────
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    meta  = get_image_meta(image, uploaded.name)

    # Image preview card
    st.markdown('<div class="img-card">', unsafe_allow_html=True)
    st.markdown('<span class="img-card-label">Preview</span>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # Classify button
    if st.button("◈  Classify Image", use_container_width=True):
        with st.spinner("Analysing..."):
            result = classify_image(image, filename=uploaded.name)

        class_name   = result["class_name"]
        pred_index   = result["prediction_index"]
        confidence   = result["confidence"]

        # Split display name into plant + condition for styled rendering
        if " — " in class_name:
            plant_part, condition_part = class_name.split(" — ", 1)
        else:
            plant_part, condition_part = class_name, ""

        # Confidence block — only show if the API returned it
        if confidence is not None:
            conf_pct = confidence * 100 if confidence <= 1.0 else confidence
            conf_html = f"""
            <div class="pred-row" style="margin-top:1.4rem;">
                <span class="pred-label" style="color:#8a8070;font-size:0.7rem">Confidence</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width:{conf_pct:.1f}%"></div>
                </div>
                <span class="pred-pct">{conf_pct:.1f}%</span>
            </div>"""
            conf_meta_html = f"""
            <div class="meta-item">
                <span class="meta-key">Confidence</span>
                <span class="meta-val">{conf_pct:.2f}%</span>
            </div>"""
        else:
            conf_html      = ""
            conf_meta_html = ""

        # Class index pill — only show if returned
        index_pill = ""
        if pred_index is not None:
            index_pill = f'<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#5a5448;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:4px;padding:2px 8px;margin-left:0.6rem;">class {pred_index}</span>'

        condition_block = f'<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#b49450;letter-spacing:0.08em;margin-top:0.3rem;">{condition_part}</div>' if condition_part else ""

        # Render results panel — use components.html for guaranteed HTML rendering
        results_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&family=Outfit:wght@200;300;400&display=swap" rel="stylesheet">
        <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: transparent; color: #e8e0d0; font-family: 'Outfit', sans-serif; font-weight: 300; }}
        .results-panel {{
            background: rgba(255,255,255,0.022);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 20px;
            padding: 2.2rem 2.4rem;
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
        }}
        .results-panel::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(180,148,80,0.4), transparent);
        }}
        .result-header {{
            font-family: 'DM Mono', monospace;
            font-size: 0.6rem;
            letter-spacing: 0.25em;
            color: #6a6058;
            text-transform: uppercase;
            margin-bottom: 1.4rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .index-pill {{
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem;
            color: #5a5448;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 4px;
            padding: 2px 8px;
        }}
        .plant-name {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.6rem;
            font-weight: 400;
            color: #f0e8d8;
            line-height: 1.1;
        }}
        .disease-name {{
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: #b49450;
            letter-spacing: 0.08em;
            margin-top: 0.35rem;
        }}
        .pred-row {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 1.4rem;
        }}
        .pred-label {{
            font-size: 0.7rem;
            color: #8a8070;
            width: 100px;
            flex-shrink: 0;
        }}
        .pred-bar-bg {{
            flex: 1;
            height: 4px;
            background: rgba(255,255,255,0.06);
            border-radius: 2px;
            overflow: hidden;
        }}
        .pred-bar-fill {{
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, #b49450, #e8c87a);
        }}
        .pred-pct {{
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            color: #7a7060;
            width: 42px;
            text-align: right;
            flex-shrink: 0;
        }}
        .meta-strip {{
            display: flex;
            gap: 2rem;
            margin-top: 2rem;
            padding-top: 1.4rem;
            border-top: 1px solid rgba(255,255,255,0.06);
            flex-wrap: wrap;
        }}
        .meta-item {{ display: flex; flex-direction: column; gap: 0.2rem; }}
        .meta-key {{
            font-family: 'DM Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.2em;
            color: #5a5448;
            text-transform: uppercase;
        }}
        .meta-val {{
            font-size: 0.82rem;
            color: #a09080;
            font-weight: 300;
        }}
        </style>
        </head>
        <body>
        <div class="results-panel">
            <div class="result-header">
                Classification Result
                {'<span class="index-pill">class ' + str(pred_index) + '</span>' if pred_index is not None else ''}
            </div>

            <div style="margin-bottom:1.4rem;">
                <div class="plant-name">{plant_part}</div>
                {'<div class="disease-name">' + condition_part + '</div>' if condition_part else ''}
            </div>

            {conf_html}

            <div class="meta-strip">
                <div class="meta-item">
                    <span class="meta-key">Filename</span>
                    <span class="meta-val">{meta['filename']}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Dimensions</span>
                    <span class="meta-val">{meta['dimensions']}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Colour Space</span>
                    <span class="meta-val">{meta['mode']}</span>
                </div>
                {conf_meta_html}
            </div>
        </div>
        </body>
        </html>
        """
        components.html(results_html, height=320, scrolling=False)

else:
    # Empty-state hint
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem 0 1rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.18em;
        color: #4a4438;
        text-transform: uppercase;
    ">
        ↑ &nbsp;&nbsp; Upload an image to begin analysis
    </div>
    """, unsafe_allow_html=True)