import os

# Giới hạn số luồng CPU để không gây nghẽn tiến trình
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import gc
from pathlib import Path

import numpy as np
import cv2 
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="AI Photo Studio 3x4", page_icon="📸", layout="wide")

st.markdown('<h2 style="text-align: center;">📸 AI PHOTO STUDIO – CHÂN DUNG 3×4 TỰ ĐỘNG</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Tách nền chuẩn • Tự nhận diện căn tỉ lệ 3×4 • Làm mịn mụn nám tự nhiên</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ CẤU HÌNH")
bg_option = st.sidebar.selectbox("🎨 Màu nền", ["Xanh dương đậm chuẩn", "Xanh ngọc sáng", "Trắng phông thẻ"])
if bg_option == "Xanh dương đậm chuẩn":
    target_bg_rgb = (0, 135, 189)
elif bg_option == "Xanh ngọc sáng":
    target_bg_rgb = (0, 168, 204)
else:
    target_bg_rgb = (255, 255, 255)

output_size_opt = st.sidebar.selectbox("📐 Kích thước xuất", ["3×4 (600×800 px)", "3×4 (900×1200 px)"])
TARGET_SIZE = (600, 800) if "600" in output_size_opt else (900, 1200)

enable_beauty = st.sidebar.checkbox("Làm sạch da & giảm mụn/nám", value=True)
smooth_level = st.sidebar.slider("Độ mịn da", 1, 5, 3)

@st.cache_resource(show_spinner=False)
def load_rembg_session():
    import onnxruntime as ort
    from rembg import new_session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    
    # Ưu tiên mô hình nhẹ u2netp / isnet để tránh lỗi tràn 1GB RAM trên Streamlit Cloud
    try:
        return new_session("isnet-general-use", sess_options=sess_opts, providers=["CPUExecutionProvider"])
    except Exception:
        return new_session("u2netp", sess_options=sess_opts, providers=["CPUExecutionProvider"])

@st.cache_resource(show_spinner=False)
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def process_background(pil_img, session_ai, bg_rgb):
    from rembg import remove
    
    w, h = pil_img.size
    # Giới hạn kích thước ảnh đầu vào khi qua AI để tiết kiệm RAM
    max_side = 1000
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        proc_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    else:
        proc_img = pil_img

    rgba = remove(proc_img, session=session_ai)
    if proc_img.size != pil_img.size:
        rgba = rgba.resize(pil_img.size, Image.Resampling.BILINEAR)

    # Ghép nền với khử mép tóc nhẹ
    np_rgba = np.array(rgba, dtype=np.float32)
    alpha = np_rgba[:, :, 3:] / 255.0
    alpha[alpha < 0.05] = 0.0
    
    rgb = np_rgba[:, :, :3]
    bg = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    
    blended = rgb * alpha + bg * (1.0 - alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def apply_skin_beauty(pil_img, intensity=3):
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]

        ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(face_mask, (w // 2, int(h * 0.45)), (int(w * 0.35), int(h * 0.45)), 0, 0, 360, 255, -1)

        combined_mask = cv2.bitwise_and(skin_mask, face_mask)
        combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)
        skin_factor = (combined_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

        d = 5 + intensity * 2
        sig = 25 + intensity * 10
        smoothed = cv2.bilateralFilter(cv_img, d=d, sigmaColor=sig, sigmaSpace=sig)
        
        weight = min(0.8, 0.4 + intensity * 0.08)
        out_bgr = (smoothed * (skin_factor * weight) + cv_img * (1.0 - skin_factor * weight)).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return pil_img

def crop_face_3x4(pil_img, bg_rgb):
    w, h = pil_img.size
    try:
        detector = load_face_cascade()
        # Chuyển ảnh nhỏ để phát hiện mặt nhanh
        small = pil_img.resize((400, int(400 * h / w)), Image.Resampling.BILINEAR)
        gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            raise ValueError("No face detected")

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        scale_ratio = w / 400.0
        fx, fy, fw, fh = [int(v * scale_ratio) for v in faces[0]]

        target_h = int(fh * 2.2)
        target_w = int(target_h * (3.0 / 4.0))

        crop_top = int(fy - (fh * 0.35) - (target_h * 0.08))
        crop_bottom = crop_top + target_h
        crop_left = int((fx + fw // 2) - target_w // 2)
        crop_right = crop_left + target_w

        # Đệm biên nếu tràn
        pad_l = max(0, -crop_left)
        pad_t = max(0, -crop_top)
        pad_r = max(0, crop_right - w)
        pad_b = max(0, crop_bottom - h)

        if pad_l or pad_t or pad_r or pad_b:
            padded = Image.new("RGB", (w + pad_l + pad_r, h + pad_t + pad_b), bg_rgb)
            padded.paste(pil_img, (pad_l, pad_t))
            return padded.crop((crop_left + pad_l, crop_top + pad_t, crop_right + pad_l, crop_bottom + pad_t))
        return pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))

    except Exception:
        # Fallback căn giữa
        target_ratio = 3.0 / 4.0
        if (w / h) > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            return pil_img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = max(0, int((h - new_h) * 0.15))
            return pil_img.crop((0, top, w, top + new_h))

# Upload
files = st.file_uploader("📁 Chọn ảnh chân dung", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if files:
    if st.button("🚀 BẮT ĐẦU XỬ LÝ ẢNH", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        session = load_rembg_session()
        
        cols = st.columns(3)
        for idx, file in enumerate(files):
            try:
                img = Image.open(io.BytesIO(file.getvalue()))
                img = ImageOps.exif_transpose(img).convert("RGB")
                
                # 1. Tách nền
                no_bg = process_background(img, session, target_bg_rgb)
                # 2. Cắt 3x4 theo khuôn mặt
                cropped = crop_face_3x4(no_bg, target_bg_rgb)
                # 3. Làm mịn da
                if enable_beauty:
                    cropped = apply_skin_beauty(cropped, smooth_level)
                
                final_out = cropped.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                final_out.save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()
                
                with cols[idx % 3]:
                    st.image(final_out, caption=file.name, use_container_width=True)
                    st.download_button("⬇️ Tải ảnh 3×4", data=img_bytes, file_name=f"3x4_{Path(file.name).stem}.jpg", mime="image/jpeg", key=f"btn_{idx}")
                
                del img, no_bg, cropped, final_out, buf
                gc.collect()
            except Exception as e:
                st.error(f"Lỗi ảnh {file.name}: {e}")
            
            progress_bar.progress((idx + 1) / len(files))
        st.success("🎉 Đã hoàn thành xử lý!")
