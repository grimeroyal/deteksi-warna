import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

st.title("Deteksi Warna dengan Area, Range HSV & Download")

st.subheader("Atur Range Warna (HSV)")

# Layout 2 kolom: slider & preset di kiri, referensi semua warna di kanan
col1, col2 = st.columns([3, 2])

with col1:
    # Inisialisasi session_state
    if "h_range" not in st.session_state:
        st.session_state.h_range = (0, 10)
    if "s_range" not in st.session_state:
        st.session_state.s_range = (120, 255)
    if "v_range" not in st.session_state:
        st.session_state.v_range = (70, 255)

    # Slider HSV (pakai session_state)
    h_min, h_max = st.slider("Hue Range", 0, 179, st.session_state.h_range, key="h_range_slider")
    s_min, s_max = st.slider("Saturation Range", 0, 255, st.session_state.s_range, key="s_range_slider")
    v_min, v_max = st.slider("Value Range", 0, 255, st.session_state.v_range, key="v_range_slider")

    # Radio button preset warna
    st.subheader("Preset Cepat")

    def set_preset():
        warna = st.session_state.warna_preset
        if warna == "Merah":
            st.session_state.h_range = (0, 10)
            st.session_state.s_range = (120, 255)
            st.session_state.v_range = (70, 255)
        elif warna == "Biru":
            st.session_state.h_range = (90, 130)
            st.session_state.s_range = (50, 255)
            st.session_state.v_range = (50, 255)
        elif warna == "Hijau":
            st.session_state.h_range = (40, 80)
            st.session_state.s_range = (40, 255)
            st.session_state.v_range = (40, 255)
        elif warna == "Kuning":
            st.session_state.h_range = (20, 30)
            st.session_state.s_range = (100, 255)
            st.session_state.v_range = (100, 255)

    st.radio(
        "Pilih preset warna:",
        ("Merah", "Biru", "Hijau", "Kuning"),
        key="warna_preset",
        on_change=set_preset
    )


with col2:
    st.markdown("### Referensi HSV Semua Warna")
    st.info("**Merah**:\nHue: 0–10 & 160–180\nSat: 120–255\nVal: 70–255")
    st.info("**Biru**:\nHue: 90–130\nSat: 50–255\nVal: 50–255")
    st.info("**Hijau**:\nHue: 40–80\nSat: 40–255\nVal: 40–255")
    st.info("**Kuning**:\nHue: 20–30\nSat: 100–255\nVal: 100–255")


# Ambil nilai slider terbaru dari session_state
h_min, h_max = st.session_state.h_range
s_min, s_max = st.session_state.s_range
v_min, v_max = st.session_state.v_range

# Slider luas area
min_area = st.slider("Minimal Luas Objek (px)", 50, 5000, 500, step=50)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Mask pakai slider/preset
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            count += 1
            cv2.drawContours(img_bgr, [c], -1, (0, 0, 0), 5)
            (x, y), radius = cv2.minEnclosingCircle(c)
            cv2.circle(img_bgr, (int(x), int(y)), int(radius), (0, 255, 0), 5)

    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Hasil Deteksi")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_result, caption=f"Jumlah objek terdeteksi: {count}")
    with col2:
        st.image(mask, caption="Mask (area terdeteksi)", clamp=True)

    st.success(f"Jumlah objek terdeteksi: {count}")

    img_pil = Image.fromarray(img_result)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Hasil Deteksi (PNG)",
        data=byte_im,
        file_name=f"hasil_deteksi.png",
        mime="image/png"
    )



