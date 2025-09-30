import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

st.title("Deteksi Warna dengan Area, Slider HSV & Download")

# Pilihan warna dasar
warna_pilihan = st.radio(
    "Pilih warna (preset):",
    ("Merah", "Biru", "Hijau", "Kuning")
)

# Preset HSV
if warna_pilihan == "Merah":
    lower_default = [0, 120, 70]
    upper_default = [10, 255, 255]
    referensi = "Merah ≈ Hue 0–10 & 160–180, S:120–255, V:70–255"
elif warna_pilihan == "Biru":
    lower_default = [90, 50, 50]
    upper_default = [130, 255, 255]
    referensi = "Biru ≈ Hue 90–130, S:50–255, V:50–255"
elif warna_pilihan == "Hijau":
    lower_default = [40, 40, 40]
    upper_default = [80, 255, 255]
    referensi = "Hijau ≈ Hue 40–80, S:40–255, V:40–255"
elif warna_pilihan == "Kuning":
    lower_default = [20, 100, 100]
    upper_default = [30, 255, 255]
    referensi = "Kuning ≈ Hue 20–30, S:100–255, V:100–255"

st.subheader("Atur Range Warna (HSV)")

# Layout 2 kolom: slider di kiri, referensi di kanan
col1, col2 = st.columns([3, 2])

with col1:
    h_min = st.slider("Hue Min", 0, 179, lower_default[0])
    s_min = st.slider("Saturation Min", 0, 255, lower_default[1])
    v_min = st.slider("Value Min", 0, 255, lower_default[2])

    h_max = st.slider("Hue Max", 0, 179, upper_default[0])
    s_max = st.slider("Saturation Max", 0, 255, upper_default[1])
    v_max = st.slider("Value Max", 0, 255, upper_default[2])

with col2:
    st.markdown("### Referensi HSV")
    st.info(referensi)

# Slider luas area
min_area = st.slider("Minimal Luas Objek (px)", 50, 5000, 500, step=50)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

# Slider HSV
h_min = st.slider("Hue Min", 0, 179, lower_default[0])
h_max = st.slider("Hue Max", 0, 179, upper_default[0])
if h_min > h_max:
    st.warning("⚠️ Hue Min tidak boleh lebih besar dari Hue Max")
    h_min, h_max = h_max, h_min  # tukar agar valid

s_min = st.slider("Saturation Min", 0, 255, lower_default[1])
s_max = st.slider("Saturation Max", 0, 255, upper_default[1])
if s_min > s_max:
    st.warning("⚠️ Saturation Min tidak boleh lebih besar dari Saturation Max")
    s_min, s_max = s_max, s_min

v_min = st.slider("Value Min", 0, 255, lower_default[2])
v_max = st.slider("Value Max", 0, 255, upper_default[2])
if v_min > v_max:
    st.warning("⚠️ Value Min tidak boleh lebih besar dari Value Max")
    v_min, v_max = v_max, v_min

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Konversi ke HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Mask pakai slider
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            count += 1
            # Boundary hitam
            cv2.drawContours(img_bgr, [c], -1, (0, 0, 0), 5)
            # Lingkaran hijau tebal
            (x, y), radius = cv2.minEnclosingCircle(c)
            cv2.circle(img_bgr, (int(x), int(y)), int(radius), (0, 255, 0), 5)

    # Konversi ke RGB untuk tampil & download
    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tampilkan hasil
    st.image(img_result,
             caption=f"Jumlah objek terdeteksi: {count}")

    st.success(f"Jumlah objek terdeteksi: {count}")

    # Tombol download hasil PNG
    img_pil = Image.fromarray(img_result)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Hasil Deteksi (PNG)",
        data=byte_im,
        file_name=f"hasil_deteksi_{warna_pilihan.lower()}.png",
        mime="image/png"
    )

