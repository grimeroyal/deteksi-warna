import cv2
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO

st.title("Deteksi Warna dengan Area, Range HSV & Download")

# Pilihan warna preset (untuk inisialisasi slider)
warna_pilihan = st.radio(
    "Pilih warna (preset awal slider):",
    ("Merah", "Biru", "Hijau", "Kuning")
)

# Preset HSV sesuai pilihan awal slider
if warna_pilihan == "Merah":
    lower_default = [0, 120, 70]
    upper_default = [10, 255, 255]
elif warna_pilihan == "Biru":
    lower_default = [90, 50, 50]
    upper_default = [130, 255, 255]
elif warna_pilihan == "Hijau":
    lower_default = [40, 40, 40]
    upper_default = [80, 255, 255]
elif warna_pilihan == "Kuning":
    lower_default = [20, 100, 100]
    upper_default = [30, 255, 255]

st.subheader("Atur Range Warna (HSV)")

# Layout 2 kolom: slider di kiri, referensi semua warna di kanan
col1, col2 = st.columns([3, 2])

with col1:
    h_min, h_max = st.slider("Hue Range", 0, 179, (lower_default[0], upper_default[0]))
    s_min, s_max = st.slider("Saturation Range", 0, 255, (lower_default[1], upper_default[1]))
    v_min, v_max = st.slider("Value Range", 0, 255, (lower_default[2], upper_default[2]))

with col2:
    st.markdown("### Referensi HSV Semua Warna")

    st.markdown("""
    | Warna  | Hue           | Saturation   | Value      |
    |--------|---------------|--------------|------------|
    | Merah  | 0–10, 160–180 | 120–255      | 70–255     |
    | Biru   | 90–130        | 50–255       | 50–255     |
    | Hijau  | 40–80         | 40–255       | 40–255     |
    | Kuning | 20–30         | 100–255      | 100–255    |
    """)

# Slider luas area
min_area = st.slider("Minimal Luas Objek (px)", 50, 5000, 500, step=50)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

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

    # Layout hasil dan mask
    st.subheader("Hasil Deteksi")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_result, caption=f"Jumlah objek terdeteksi: {count}")
    with col2:
        st.image(mask, caption="Mask (area terdeteksi)", clamp=True)

    st.success(f"Jumlah objek terdeteksi: {count}")

    # Tombol download hasil PNG
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






