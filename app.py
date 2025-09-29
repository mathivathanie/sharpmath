# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import time, datetime, os, random

# ----- TFLite availability check -----
tflite_available = True
try:
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import Interpreter
except Exception:
    tflite_available = False

# ----- Page config & minimal styling -----
st.set_page_config(page_title="NutriLens — Demo", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.page {max-width:650px; margin:auto;}
.title {font-size:24px; font-weight:700; margin-bottom:4px;}
.subtitle {color:#6b7280; margin-bottom:12px;}
.card {background: linear-gradient(180deg,#ffffff,#fbfdff); padding:12px; border-radius:12px; box-shadow:0 6px 24px rgba(16,24,40,0.06);}
.big-btn {display:inline-block; padding:12px 18px; border-radius:10px; font-weight:600; text-decoration:none;}
.small-muted {color:#6b7280; font-size:13px;}
@media (max-width:420px) {.title{font-size:20px}}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="page">', unsafe_allow_html=True)
st.markdown('<div class="title">🍽️ NutriLens — Prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Point your phone camera at the plate • get instant nutrition</div>', unsafe_allow_html=True)

# ----- Load TFLite model -----
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
                raw_labels = [line.strip().lower() for line in f.readlines()]
                labels = []
                for lbl in raw_labels:
                    parts = lbl.split(maxsplit=1)
                    name = parts[1] if len(parts) > 1 else parts[0]
                    if name == "pomogranate":
                        name = "pomegranate"
                    labels.append(name)
    except Exception:
        interpreter = None

# ----- Nutrition DB -----
NUTRITION_DB = {
    "apple": {"cal": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "fiber": 4},
    "half apple": {"cal": 48, "protein": 0.25, "carbs": 12.5, "fat": 0.15, "fiber": 2},
    "pomegranate": {"cal": 105, "protein": 1.7, "carbs": 26, "fat": 1.2, "fiber": 4},
    "goodday": {"cal": 120, "protein": 1.5, "carbs": 18, "fat": 5, "fiber": 1},
    "samosa": {"cal": 262, "protein": 4.3, "carbs": 30, "fat": 14, "fiber": 2},
    "bingo": {"cal": 160, "protein": 2, "carbs": 15, "fat": 10, "fiber": 1},
}
HEALTHY = {"apple", "pomegranate", "half apple"}
JUNK = set(NUTRITION_DB.keys()) - HEALTHY

# ----- Session state -----
if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False
if "daily_quota" not in st.session_state:
    st.session_state.daily_quota = None
if "remaining" not in st.session_state:
    st.session_state.remaining = None
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

# ----- Profile area -----
if not st.session_state.profile_saved:
    with st.form("profile_form", clear_on_submit=False):
        cols = st.columns([1,1,1])
        with cols[0]:
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
        with cols[1]:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=180, value=60)
        with cols[2]:
            age = st.number_input("Age", min_value=10, max_value=100, value=22)
        saved = st.form_submit_button("Save profile")
        if saved:
            total = max(1200, int(weight * 30))
            st.session_state.daily_quota = total
            st.session_state.remaining = total
            st.session_state.profile_saved = True
            st.success(f"Profile saved · daily target ≈ {total} kcal")

# Main card
st.markdown('<div class="card">', unsafe_allow_html=True)
cols = st.columns([2,1])
with cols[0]:
    captured_file = st.camera_input("Tap to capture (recommended)")
    uploaded_file = st.file_uploader("Or upload image", type=["jpg","jpeg","png"])
with cols[1]:
    st.markdown("**Progress**")
    if st.session_state.daily_quota:
        used = st.session_state.daily_quota - (st.session_state.remaining or 0)
        pct = int((used / st.session_state.daily_quota) * 100) if st.session_state.daily_quota else 0
        st.progress(min(max(pct, 0), 100))
        st.metric("Remaining kcal", f"{st.session_state.remaining} kcal")
        lvl = 1 + (st.session_state.xp // 50)
        st.metric("XP", f"{st.session_state.xp} (Level {lvl})")
    else:
        st.info("Save profile to enable tracking", icon="ℹ️")
st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ----- helper: run tflite predict -----
def tflite_predict(pil_img: Image.Image):
    if interpreter is None:
        return None
    target = (224, 224)
    try:
        img = ImageOps.fit(pil_img, target, Image.Resampling.LANCZOS).convert("RGB")
        arr = np.array(img) / 255.0
        input_data = np.expand_dims(arr.astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]
        idx = int(np.argmax(preds))
        label = labels[idx] if labels else str(idx)
        conf = float(preds[idx])
        return label.lower(), conf
    except Exception:
        return None

# fallback classifier
def fallback_classify(fileobj):
    name = getattr(fileobj, "name", "") or ""
    name = name.lower()
    for k in NUTRITION_DB:
        if k in name:
            return k, 0.9
    return "apple", 0.6

# ----- Process capture / upload -----
image_to_process = captured_file or uploaded_file

if image_to_process and not st.session_state.profile_saved:
    st.warning("Save your profile first so we can track your daily calories.")
elif image_to_process:
    st.image(Image.open(image_to_process).convert("RGB"), caption="Preview", use_column_width=True)
    with st.spinner("Analyzing..."):
        time.sleep(0.5)

    predicted = None
    if interpreter is not None and labels:
        res = tflite_predict(Image.open(image_to_process))
        if res:
            predicted = res
    if predicted is None:
        predicted = fallback_classify(image_to_process)
    label, conf = predicted
    label = label.lower()
    st.session_state.last_scan = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"### Detected: **{label.capitalize()}**  —  ({conf:.2f})")

    if label in NUTRITION_DB:
        d = NUTRITION_DB[label]
        cols = st.columns([1,1])
        with cols[0]:
            st.markdown(f"**Calories**\n\n# {d['cal']} kcal")
            st.markdown(f"**Protein**: {d['protein']} g")
        with cols[1]:
            st.markdown(f"**Carbs**: {d['carbs']} g")
            st.markdown(f"**Fat**: {d['fat']} g")
            st.markdown(f"**Fiber**: {d['fiber']} g")

        prev = st.session_state.remaining if st.session_state.remaining is not None else st.session_state.daily_quota
        st.session_state.remaining = prev - d["cal"]
        if label in JUNK:
            st.session_state.xp = max(0, st.session_state.xp - 5)
        else:
            st.session_state.xp += 12

        if st.session_state.daily_quota:
            used = st.session_state.daily_quota - st.session_state.remaining
            percent = int((used / st.session_state.daily_quota) * 100)
            st.progress(min(100, max(0, percent)))
            st.write(f"Remaining today: **{st.session_state.remaining} kcal**")

    else:
        st.write("Nutrition data not available for this item.")
        st.session_state.history.append({
            "time": st.session_state.last_scan, "food": label, "cal": None, "remaining": st.session_state.remaining
        })

    # Time-aware tip
    now = datetime.datetime.now()
    hour = now.hour
    if hour < 11:
        tod = "morning"; suggested = "Try a protein-rich breakfast or fruit later."
    elif hour < 15:
        tod = "afternoon"; suggested = "A balanced lunch with veggies and protein is great now."
    elif hour < 19:
        tod = "evening"; suggested = "Light dinner recommended — prefer salads or soup."
    else:
        tod = "night"; suggested = "It's late — keep choices light and low-cal."

    if label in JUNK:
        quotes = [
            "Crunchy now, couch-bound later — consider a swap next time! 😅",
            "Taste-bud party today, fitness party tomorrow — balance it out!",
            "Cheat day accepted — don't sign a lifetime deal with junk! 😄"
        ]
        st.error("⚠️ Junk Food Detected", icon="⚠️")
        st.write(f"**{random.choice(quotes)}**")
        st.info(f"Tip for the {tod}: {suggested}")
        st.snow()
    else:
        st.success("✅ Good choice — this helps your goals!", icon="✅")
        tips = ["Add a small salad for fiber next time.", "Great — stay consistent!", "Hydrate well after this meal."]
        st.write(f"**Tip:** {random.choice(tips)}")
        st.info(f"Tip for the {tod}: {suggested}")
        st.balloons()

    st.session_state.history.append({
        "time": st.session_state.last_scan,
        "food": label,
        "cal": NUTRITION_DB.get(label, {}).get("cal"),
        "remaining": st.session_state.remaining
    })

    st.markdown("### Recent")
    for h in reversed(st.session_state.history[-5:]):
        cal_text = f"{h['cal']} kcal" if h.get("cal") else "—"
        st.write(f"- {h['time'].split(' ')[1]}  •  {h['food'].capitalize()}  •  {cal_text}  •  remaining {h['remaining']}")

# Instructions if nothing captured
if not image_to_process and st.session_state.profile_saved:
    st.markdown("<div class='small-muted'>Tap the camera button above to simulate the smart-glasses scan.</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>Demo: works best on mobile browser. Keep model.tflite & labels.txt in the app folder for live classification.</div>", unsafe_allow_html=True)
if interpreter is None:
    st.warning("TFLite model not loaded — app using fallback. Place model.tflite and labels.txt next to app.py and refresh to enable the full demo.", icon="⚠️")

st.markdown('</div>', unsafe_allow_html=True)
