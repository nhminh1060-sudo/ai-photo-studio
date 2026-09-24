import os

# ============================================================
# CẤU HÌNH GIỚI HẠN LUỒNG TRÊN WINDOWS (CHỐNG TREO MÁY)
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import gc
import shutil
import zipfile
from pathlib import Path

import numpy as np
import cv2 
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="AI Photo Studio - Tách Nền, Auto Crop 3×4 & Làm Mịn Da", page_icon="📸", layout="wide")

st.markdown("""
    <style>
        .main-title { font-size: 28px; font-weight: 700; margin-bottom: 5px; }
        .sub-title { color: #555; font-size: 15px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📸 AI PHOTO STUDIO – CHÂN DUNG 3×4 STUDIO TOÀN DIỆN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tách nền tóc tơ • Tự nhận diện & cắt 3×4 chuẩn thẻ • Tự động xóa mụn, nám, làm sáng mịn da tự nhiên</div>', unsafe_allow_html=True)

# ============================================================
# CẤU HÌNH SIDEBAR
# ============================================================
st.sidebar.header("⚙️ CẤU HÌNH ẢNH THẺ")
bg_color_option = st.sidebar.selectbox("🎨 Chọn màu nền", ["Xanh dương đậm chuẩn", "Xanh ngọc sáng", "Trắng phông thẻ"])
if bg_color_option == "Xanh dương đậm chuẩn":
    target_bg_rgb = (0, 135, 189)
elif bg_color_option == "Xanh ngọc sáng":
    target_bg_rgb = (0, 168, 204)
else:
    target_bg_rgb = (255, 255, 255)

output_size_option = st.sidebar.selectbox("📐 Kích thước ảnh xuất", ["3×4 – 600×800 px", "3×4 – 900×1200 px"])
TARGET_SIZE = (600, 800) if "600" in output_size_option else (900, 1200)

st.sidebar.markdown("---")
st.sidebar.header("✨ LÀM ĐẸP DA TỰ ĐỘNG")
enable_retouch = st.sidebar.checkbox("Bật tự động làm sạch da & giảm mụn/nám", value=True)
smooth_intensity = st.sidebar.slider("Mức độ mịn da tự nhiên", min_value=1, max_value=5, value=3, help="Mức 2-3 là đẹp chuẩn studio, giữ nét tự nhiên chân thật.")

# ============================================================
# CẤU HÌNH MÔ HÌNH CHÂN DUNG BIREFNET
# ============================================================
MODEL_NAME = "birefnet-portrait"
APP_DIR = Path(__file__).resolve().parent
u2net_dir = Path.home() / ".u2net"
u2net_dir.mkdir(parents=True, exist_ok=True)

target_model_path = u2net_dir / f"{MODEL_NAME}.onnx"
local_model_path = APP_DIR / f"{MODEL_NAME}.onnx"

if not target_model_path.exists() and local_model_path.exists():
    try:
        shutil.copy2(local_model_path, target_model_path)
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def get_portrait_session():
    import onnxruntime as ort
    from rembg import new_session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    try:
        return new_session("birefnet-portrait", sess_options=sess_opts, providers=["CPUExecutionProvider"])
    except Exception:
        try:
            return new_session("isnet-general-use", sess_options=sess_opts, providers=["CPUExecutionProvider"])
        except Exception:
            return new_session("u2net_human_seg", sess_options=sess_opts, providers=["CPUExecutionProvider"])

@st.cache_resource(show_spinner=False)
def get_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector

def load_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def studio_alpha_matte_blend(rgba_img, bg_rgb):
    np_img = np.array(rgba_img, dtype=np.float32)
    rgb = np_img[:, :, :3]
    alpha = np_img[:, :, 3] / 255.0
    h, w = alpha.shape

    alpha[alpha < 0.08] = 0.0

    fringe_zone = (alpha > 0.08) & (alpha < 0.85)
    head_mask = np.zeros((h, w), dtype=bool)
    head_mask[:int(h * 0.70), :] = True
    target_hair = fringe_zone & head_mask

    suppression_factor = np.expand_dims(np.where(target_hair, np.power(alpha, 0.4), 1.0), axis=-1)
    rgb_corrected = rgb * suppression_factor

    bg_color = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    alpha_3d = np.expand_dims(alpha, axis=-1)

    final_rgb = rgb_corrected * alpha_3d + bg_color * (1.0 - alpha_3d)
    final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)

    return Image.fromarray(final_rgb, "RGB")

def compose_background(image, session_ai, bg_rgb):
    from rembg import remove
    
    orig_size = image.size
    max_dim = 1400
    w, h = orig_size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        small_image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:
        small_image = image

    output_small = remove(small_image, session=session_ai)
    output_rgba = output_small.resize(orig_size, Image.Resampling.LANCZOS)

    return studio_alpha_matte_blend(output_rgba, bg_rgb)

# ============================================================
# THUẬT TOÁN TỰ ĐỘNG XÓA MỤN, NÁM VÀ LÀM MỊN DA STUDIO
# ============================================================
def auto_skin_beauty_retouch(pil_img, intensity=3):
    """
    Thuật toán làm sạch da tự động:
    1. Nhận diện vùng da qua không gian YCrCb và vùng elip mặt.
    2. Áp dụng Edge-Preserving Filter triệt tiêu mụn, tàn nhang, thâm nám.
    3. Giữ nguyên chi tiết sắc nét của mắt, lông mày, môi và tóc.
    """
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]

        # 1. Phát hiện vùng da người (Skin Segmentation qua dải YCrCb)
        ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # 2. Định vị khuôn mặt để tập trung làm đẹp vùng mặt và cổ
        detector = get_face_detector()
        face_roi_mask = np.zeros((h, w), dtype=np.uint8)
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = []
        if detector is not None:
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        if len(faces) > 0:
            # Chọn mặt chính
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            # Tạo vùng elip bao phủ khuôn mặt và mở rộng xuống phần cổ
            center = (fx + fw // 2, int(fy + fh * 0.6))
            axes = (int(fw * 0.65), int(fh * 0.85))
            cv2.ellipse(face_roi_mask, center, axes, 0, 0, 360, 255, -1)
        else:
            # Nếu không tìm thấy tọa độ cụ thể, lấy vùng trung tâm ảnh
            cv2.ellipse(face_roi_mask, (w // 2, int(h * 0.45)), (int(w * 0.35), int(h * 0.45)), 0, 0, 360, 255, -1)

        # Kết hợp Mask da và Mask vùng mặt
        effective_skin_mask = cv2.bitwise_and(skin_mask, face_roi_mask)

        # Làm mềm mép mask da để hiệu ứng chuyển tiếp tự nhiên
        effective_skin_mask = cv2.GaussianBlur(effective_skin_mask, (21, 21), 0)
        skin_factor = (effective_skin_mask.astype(np.float32) / 255.0)

        # 3. Lọc song phương (Bilateral Filter) làm mờ mụn, thâm, nám nhưng giữ sắc cạnh
        diameter = 5 + intensity * 2
        sigma_color = 20 + intensity * 10
        sigma_space = 20 + intensity * 10
        smooth_bgr = cv2.bilateralFilter(cv_img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # Tăng sáng nhẹ tone da (~3%) để da tươi sáng tự nhiên
        smooth_bgr = cv2.convertScaleAbs(smooth_bgr, alpha=1.03, beta=2)

        # 4. Hòa trộn lớp mịn với ảnh gốc theo mặt nạ da
        skin_factor_3d = np.repeat(skin_factor[:, :, np.newaxis], 3, axis=2)
        # Tỷ lệ hòa trộn: giữ lại một chút kết cấu thật tránh bị bệt như sáp
        blend_weight = min(0.85, 0.45 + intensity * 0.08)
        retouched_bgr = (smooth_bgr * (skin_factor_3d * blend_weight) + 
                         cv_img * (1.0 - skin_factor_3d * blend_weight)).astype(np.uint8)

        retouched_rgb = cv2.cvtColor(retouched_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(retouched_rgb)
    except Exception:
        return pil_img

# ============================================================
# TỰ ĐỘNG CẮT 3x4 THEO KHUÔN MẶT
# ============================================================
def fallback_crop_3x4(pil_img):
    w, h = pil_img.size
    target_ratio = 3.0 / 4.0
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return pil_img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = int((h - new_h) * 0.15)
        top = max(0, min(top, h - new_h))
        return pil_img.crop((0, top, w, top + new_h))

def auto_crop_face_3x4(pil_img, bg_rgb):
    try:
        detector = get_face_detector()
        if detector is None:
            return fallback_crop_3x4(pil_img)

        img_w, img_h = pil_img.size
        det_max = 800
        scale = 1.0
        if max(img_w, img_h) > det_max:
            scale = det_max / float(max(img_w, img_h))
            small_w, small_h = int(img_w * scale), int(img_h * scale)
            detect_img = pil_img.resize((small_w, small_h), Image.Resampling.BILINEAR)
        else:
            detect_img = pil_img

        cv_img = cv2.cvtColor(np.array(detect_img), cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

        if len(faces) == 0:
            return fallback_crop_3x4(pil_img)

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        
        fx = int(fx / scale)
        fy = int(fy / scale)
        fw = int(fw / scale)
        fh = int(fh / scale)

        center_x = fx + fw // 2
        target_crop_h = int(fh * 2.2)
        target_crop_w = int(target_crop_h * (3.0 / 4.0))

        crop_top = int(fy - (fh * 0.35) - (target_crop_h * 0.08))
        crop_bottom = crop_top + target_crop_h
        crop_left = int(center_x - target_crop_w // 2)
        crop_right = crop_left + target_crop_w

        pad_left = max(0, -crop_left)
        pad_top = max(0, -crop_top)
        pad_right = max(0, crop_right - img_w)
        pad_bottom = max(0, crop_bottom - img_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_w = img_w + pad_left + pad_right
            padded_h = img_h + pad_top + pad_bottom
            padded_img = Image.new("RGB", (padded_w, padded_h), bg_rgb)
            padded_img.paste(pil_img, (pad_left, pad_top))
            
            crop_left += pad_left
            crop_top += pad_top
            crop_right += pad_left
            crop_bottom += pad_top
            return padded_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        else:
            return pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))

    except Exception:
        return fallback_crop_3x4(pil_img)

def finalize_image(image, target_size, bg_rgb, do_retouch=True, intensity=3):
    # Bước 1: Cắt ảnh chuẩn tỉ lệ 3x4 theo vị trí đầu và vai
    image = auto_crop_face_3x4(image, bg_rgb)
    
    # Bước 2: Tự động làm sạch da, mờ thâm nám mụn (thực hiện sau khi crop để xử lý đúng tỉ lệ mặt)
    if do_retouch:
        image = auto_skin_beauty_retouch(image, intensity=intensity)

    # Bước 3: Resize theo độ phân giải người dùng đã chọn
    return image.resize(target_size, Image.Resampling.LANCZOS)

def image_to_jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100, subsampling=0, optimize=True)
    return buffer.getvalue()

# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
uploaded_files = st.file_uploader(
    "📁 Chọn ảnh học sinh / giáo viên", 
    type=["jpg", "jpeg", "png", "webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📌 Đã nạp **{len(uploaded_files)} ảnh**.")
    process_button = st.button("🚀 BẮT ĐẦU XỬ LÝ (TÁCH NỀN • AUTO CROP 3×4 • MỊN DA)", type="primary", use_container_width=True)

    if process_button:
        status_text = st.empty()
        progress_bar = st.progress(0)
        processed_images = []
        errors = []

        try:
            status_text.info("⏳ Đang chuẩn bị động cơ AI & bộ nhận diện...")
            session = get_portrait_session()
            status_text.success("✅ Động cơ AI sẵn sàng!")

            total = len(uploaded_files)
            for index, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.info(f"⚙️ Đang xử lý **{index + 1}/{total}**: `{filename}` (Tách nền • Cắt 3×4 • Làm sạch da)")

                try:
                    input_bytes = uploaded_file.getvalue()
                    original = load_image(input_bytes)
                    
                    # 1. Bóc tách nền và ghép phông xanh studio
                    processed = compose_background(original, session, target_bg_rgb)
                    
                    # 2. Cắt 3x4 tự động + Làm sạch da mụn nám
                    final_image = finalize_image(
                        processed, 
                        TARGET_SIZE, 
                        target_bg_rgb, 
                        do_retouch=enable_retouch, 
                        intensity=smooth_intensity
                    )
                    final_bytes = image_to_jpeg_bytes(final_image)
                    
                    stem = Path(filename).stem
                    output_name = f"3x4_{stem}.jpg"
                    processed_images.append((output_name, final_image.copy(), final_bytes))
                    del original, processed, final_image, input_bytes
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")

                progress_bar.progress((index + 1) / total)
                gc.collect()

            if processed_images:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for output_name, _image, output_bytes in processed_images:
                        zip_file.writestr(output_name, output_bytes)
                zip_data = zip_buffer.getvalue()

                status_text.success(f"🎉 Hoàn thành! Đã xử lý xong **{len(processed_images)}/{total} ảnh** chuẩn Studio.")
                st.download_button(
                    "📦 TẢI TOÀN BỘ ẢNH HOÀN CHỈNH (ZIP)", 
                    data=zip_data, 
                    file_name="Anh_The_Studio_3x4_SkinRetouch.zip", 
                    mime="application/zip", 
                    type="primary", 
                    use_container_width=True
                )

                st.markdown("---")
                columns = st.columns(3)
                for index, (output_name, preview_image, output_bytes) in enumerate(processed_images):
                    with columns[index % 3]:
                        st.image(preview_image, caption=output_name, use_container_width=True)
                        st.download_button(
                            "⬇️ Tải ảnh này", 
                            data=output_bytes, 
                            file_name=output_name, 
                            mime="image/jpeg", 
                            key=f"dl_{index}", 
                            use_container_width=True
                        )
            else:
                status_text.empty()
                st.error("❌ Không có ảnh nào được xử lý thành công.")

            if errors:
                with st.expander("🔎 Xem chi tiết lỗi"):
                    for error in errors:
                        st.write("• " + error)

        except Exception as e:
            st.error("❌ Có lỗi trong quá trình xử lý.")
            st.exception(e)
        finally:
            gc.collect()