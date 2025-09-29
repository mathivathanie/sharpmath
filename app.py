# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import time, datetime, os
import random

# Try to import tflite; fall back gracefully if not present
tflite_available = True
try:
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import Interpreter
except Exception as e:
    tflite_available = False

st.set_page_config(page_title="NutriLens - Demo", layout="centered", initial_sidebar_state="collapsed")

# ----------------------------
# Helper: styling (simple CSS)
# ----------------------------
st.markdown(
    """
    <style>
    .big-title {font-size:28px; font-weight:600;}
    .muted {color: #6c757d; font-size:14px;}
    .card {background: linear-gradient(90deg, #ffffff, #f7f9fc); padding:14px; border-radius:12px; box-shadow: 0 4px 20px rgba(20,30,50,0.06);}
    .small {font-size:13px;}
    .food-name {font-size:20px; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🍽️ NutriLens — Mobile Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Smart-glasses simulation using phone camera — demo-ready prototype</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Load model if present
# ----------------------------
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
interpreter = None
labels = []

if tflite_available and os.path.exists(MODEL_PATH):
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                labels = [line.strip() for line in f.readlines()]
        else:
            labels = []
    except Exception as e:
        st.warning("Model found but couldn't be loaded. Falling back to simple demo mode.")
        interpreter = None
else:
    interpreter = None

# ----------------------------
# Nutrition database for demo (includes user's items)
# ----------------------------
# Values are approximate — demo values
NUTRITION_DB = {
    "goodday": {"cal": 120, "protein": 1.5, "carbs": 18, "fat": 5, "fiber": 1},
    "banana chips": {"cal": 150, "protein": 1.8, "carbs": 8, "fat": 12, "fiber": 1},
    "apple": {"cal": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "fiber": 4},
    "bingo": {"cal": 140, "protein": 2, "carbs": 16, "fat": 8, "fiber": 1},
    "kurkure": {"cal": 160, "protein": 2, "carbs": 15, "fat": 9, "fiber": 1},
    # fallback examples
    "dosa": {"cal":133, "protein":2.7, "carbs":25, "fat":3, "fiber":1}
}

JUNK_SET = {"goodday", "banana chips", "bingo", "kurkure"}  # junk categories

# ----------------------------
# Session state initialization
# ----------------------------
if "daily_quota" not in st.session_state:
    st.session_state.daily_quota = None
if "remaining" not in st.session_state:
    st.session_state.remaining = None
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts with entries

# ----------------------------
# Sidebar: profile entry
# ----------------------------
with st.expander("👤 User profile & settings", expanded=True):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=180, value=60)
    with col3:
        age = st.number_input("Age", min_value=10, max_value=100, value=22)

    if st.button("Save profile"):
        # Simple calorie estimate: Mifflin-like roughness or weight*30
        daily = int(weight * 30)  # simple for demo
        st.session_state.daily_quota = daily
        # if remaining already set, keep it; else initialize
        if st.session_state.remaining is None:
            st.session_state.remaining = daily
        st.success(f"Profile saved. Estimated daily calories ≈ {daily} kcal")

st.write("")

# ----------------------------
# Main UI: Camera / upload
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🔍 Scan a plate (use camera or upload an image)")

colA, colB = st.columns([2,1])
with colA:
    # camera_input works on mobile browsers
    captured_file = st.camera_input("Point camera and capture (recommended for demo)")
    uploaded_file = st.file_uploader(" — or upload an image", type=["jpg","jpeg","png"])
with colB:
    # Gamified progress + daily bar
    st.markdown("**Daily progress**")
    if st.session_state.daily_quota:
        used = max(0, st.session_state.daily_quota - (st.session_state.remaining or 0))
        prog = int(used / st.session_state.daily_quota * 100) if st.session_state.daily_quota else 0
        st.progress(prog if prog <= 100 else 100)
        st.metric("Remaining kcal", f"{st.session_state.remaining} kcal" if st.session_state.remaining is not None else "—")
        st.metric("XP", f"{st.session_state.xp}")
    else:
        st.info("Set profile to enable daily tracking")

st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Utility: predict function using TFLite (if available)
# ----------------------------
def tflite_predict(pil_img: Image.Image):
    if interpreter is None:
        return None
    # resize according to common TM size 224x224 - many TM models use 224
    target = (224, 224)
    img = ImageOps.fit(pil_img, target, Image.ANTIALIAS).convert("RGB")
    arr = np.array(img) / 255.0
    input_data = np.expand_dims(arr.astype(np.float32), axis=0)
    try:
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]
        idx = int(np.argmax(preds))
        label = labels[idx] if labels else str(idx)
        confidence = float(preds[idx])
        return label, confidence
    except Exception as e:
        return None

# fallback naive classification by filename heuristics
def fallback_classify(filename_or_image):
    s = ""
    if isinstance(filename_or_image, str):
        s = filename_or_image.lower()
    else:
        # try to extract name from file object if available
        try:
            s = getattr(filename_or_image, "name", "") or ""
            s = s.lower()
        except:
            s = ""
    # simple keyword mapping
    for key in NUTRITION_DB.keys():
        if key in s:
            return key, 0.9
    # if nothing, pick apple as safe default
    return "apple", 0.6

# ----------------------------
# Processing: when an image is provided
# ----------------------------
image_to_process = None
if captured_file is not None:
    image_to_process = captured_file
elif uploaded_file is not None:
    image_to_process = uploaded_file

if image_to_process and st.session_state.daily_quota is None:
    st.warning("Please save your profile (height & weight) first so we can track daily calories.")
elif image_to_process:
    # Show preview and simulate 3-second gaze
    st.info("Hold steady... Gazing for 3 seconds to simulate smart-glasses capture.")
    preview = Image.open(image_to_process).convert("RGB")
    st.image(preview, caption="Preview", use_column_width=True)

    # "gaze" countdown
    with st.spinner("Analyzing (3s)..."):
        for i in range(3, 0, -1):
            st.write(f"Gaze: {i}…")
            time.sleep(1)

    # Predict via model if available
    result = None
    if interpreter is not None and labels:
        res = tflite_predict(preview)
        if res:
            detected, conf = res
            # normalize label names to lowercase simple matching to db keys
            detected_key = detected.strip().lower()
            # sometimes labels may have spaces or capital letters; try matching known DB keys
            # attempt small heuristics
            mapping = {k: k for k in NUTRITION_DB.keys()}
            # if detected not in db keys, attempt to find closest match
            if detected_key not in NUTRITION_DB:
                # try to find a key that is substring
                for k in NUTRITION_DB:
                    if k in detected_key or detected_key in k:
                        detected_key = k
                        break
            result = (detected_key, conf)
    # if model not present or prediction failed, fallback
    if result is None:
        fallback = fallback_classify(getattr(image_to_process, "name", "") or "")
        result = fallback

    food_label, confidence = result
    food_label = food_label.lower()
    st.success(f"Detected: **{food_label.capitalize()}**  (confidence: {confidence:.2f})")

    # lookup nutrition
    if food_label in NUTRITION_DB:
        d = NUTRITION_DB[food_label]
        # show details in nice layout
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Calories**\n\n# {d['cal']} kcal")
            st.markdown(f"**Protein**: {d['protein']} g")
        with col2:
            st.markdown(f"**Carbs**: {d['carbs']} g")
            st.markdown(f"**Fat**: {d['fat']} g")
            st.markdown(f"**Fiber**: {d['fiber']} g")
        st.markdown("---")

        # Decrement remaining calories and award XP
        prev_remaining = st.session_state.remaining if st.session_state.remaining is not None else st.session_state.daily_quota
        if prev_remaining is None:
            prev_remaining = st.session_state.daily_quota

        new_remaining = prev_remaining - d["cal"]
        st.session_state.remaining = new_remaining
        # Award XP for healthy choices, less or negative XP for junk
        if food_label in JUNK_SET:
            xp_change = -5
        else:
            xp_change = 10
        st.session_state.xp = max(0, st.session_state.xp + xp_change)

        # Save history
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "food": food_label,
            "cal": d["cal"],
            "remaining": new_remaining
        })

        # Show remaining and progress bar
        if st.session_state.daily_quota:
            used = st.session_state.daily_quota - st.session_state.remaining
            percent = int((used / st.session_state.daily_quota) * 100)
            st.progress(min(100, max(0, percent)))
            st.write(f"Remaining calories today: **{st.session_state.remaining} kcal**")
    else:
        st.warning("Nutrition data not available for this item. It will be added to history as unknown.")
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "food": food_label,
            "cal": None,
            "remaining": st.session_state.remaining
        })

    # Time-aware suggestions and motivational quotes
    now = datetime.datetime.now()
    hour = now.hour
    if hour < 11:
        time_of_day = "morning"
        suggested = "a protein-rich breakfast or fruits later."
    elif hour < 15:
        time_of_day = "afternoon"
        suggested = "a balanced lunch with veggies and protein."
    elif hour < 19:
        time_of_day = "evening"
        suggested = "a light dinner — prefer salads or soup."
    else:
        time_of_day = "night"
        suggested = "light and low-cal choices; avoid heavy meals before sleep."

    if food_label in JUNK_SET:
        # funny motivational quotes pool
        quotes = [
            "Crunchy now, couch-bound later — maybe swap next time? 😅",
            "Your taste buds just threw a party. Your goals RSVP'd 'maybe' 🤷‍♂️",
            "It's okay to cheat — just don't change your name to 'Always Cheat' 😀",
            "Junk today, jog tomorrow — we've all been there. Try a fruit next!"
        ]
        q = random.choice(quotes)
        st.error("⚠️ Junk Food Detected")
        st.write(f"**{q}**")
        st.info(f"Tip for the {time_of_day}: {suggested}")
        # playful animation
        st.write("")
        st.snow()
    else:
        st.success("✅ Nice choice! This helps your goals.")
        tips = [
            "Try adding a small bowl of salad next time for fiber boost.",
            "Great! Balance with greens and water.",
            "Keep it up — consistency > intensity."
        ]
        st.write(f"**Tip:** {random.choice(tips)}")
        st.info(f"Tip for the {time_of_day}: {suggested}")
        # small celebration
        st.balloons()

    # show recent history compact
    st.markdown("### 🕘 Recent entries")
    for h in reversed(st.session_state.history[-5:]):
        hh = f"- {h['time'].split(' ')[1]} • {h['food'].capitalize()} • {h['cal'] or '—'} kcal • remaining {h['remaining']}"
        st.write(hh)

# ----------------------------
# Footer: fallback instruction & debug
# ----------------------------
st.markdown("---")
st.markdown("#### Demo notes")
st.markdown(
    "- Use **camera capture** on your phone for the most realistic demo (Streamlit's `camera_input`).\n"
    "- If the model fails to load, the app will use filename heuristics (so name images like `banana chips.jpg`).\n"
    "- Keep `model.tflite` and `labels.txt` in the same folder as `app.py` for full automation.\n"
)
if interpreter is None:
    st.warning("TFLite model not loaded. App is using fallback classification. To load model, place a valid `model.tflite` and `labels.txt` here and refresh.")
