import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("Deteksi Warna dengan Area & Download")

# Pilihan warna
warna_pilihan = st.radio(
    "Pilih warna:",
    ("Merah", "Biru", "Hijau", "Kuning")
)

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

    # Range warna preset
    if warna_pilihan == "Merah":
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 120, 70])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    elif warna_pilihan == "Biru":
        lower = np.array([90, 50, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif warna_pilihan == "Hijau":
        lower = np.array([40, 40, 40])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif warna_pilihan == "Kuning":
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
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
             caption=f"Jumlah objek {warna_pilihan.lower()} terdeteksi: {count}")

    st.success(f"Jumlah objek {warna_pilihan.lower()} terdeteksi: {count}")

    # Tombol download hasil PNG
    from io import BytesIO
    import base64

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
