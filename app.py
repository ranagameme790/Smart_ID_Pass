import os
import re
import base64
import sqlite3
import cv2
import numpy as np
import bidi.algorithm
import bidi
bidi.get_display = bidi.algorithm.get_display
import easyocr
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify

from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'smart-id-secret-2026'

# الإعدادات
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
UPLOAD_FOLDER = 'static/uploads'

# إنشاء مجلد الرفع إن لم يكن موجوداً
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

DATABASE_PATH = os.path.join(os.getcwd(), 'student_face_database.sqlite')
LIVENESS_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.5

_face_cascade = None
_face_analyzer = None
_ocr_reader = None

def allowed_file(filename):
    """التحقق من أن الملف مسموح به"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_face_detector():
    global _face_cascade
    if _face_cascade is None:
        # Use OpenCV's built-in Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False)
    return _ocr_reader


def get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalysis(name='buffalo_l')
        _face_analyzer.prepare(ctx_id=0)
    return _face_analyzer


def predict_liveness(image_path):
    """Enhanced liveness detection to prevent photo attacks while allowing mobile camera photos."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Error in reading selfie for liveness check')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_detector()

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return {
            'is_liveness': False,
            'confidence': 0,
            'raw_score': 0.0
        }

    # Use the largest face detected
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]

    # Extract face region
    face_roi = gray[y:y+h, x:x+w]

    # 1. Check face size relative to image (relaxed for mobile cameras)
    face_area = w * h
    image_area = img.shape[0] * img.shape[1]
    face_ratio = face_area / image_area

    if face_ratio < 0.02:  # Relaxed from 0.05 to 0.02 for mobile photos
        return {
            'is_liveness': False,
            'confidence': 0,
            'raw_score': 0.0
        }

    # 2. Check for photo characteristics (uniform lighting, lack of natural variation)
    # Calculate variance of pixel intensities in face region
    face_variance = np.var(face_roi.astype(np.float32))

    # Photos typically have lower variance than live faces, but mobile photos can also be low
    if face_variance < 50:  # Relaxed from 100 to 50
        return {
            'is_liveness': False,
            'confidence': 0,
            'raw_score': 0.0
        }

    # 3. Check for natural skin texture using Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()

    if laplacian_var < 20:  # Relaxed from 50 to 20 for mobile camera quality
        return {
            'is_liveness': False,
            'confidence': 0,
            'raw_score': 0.0
        }

    # 4. Check for color variation (photos often have less color depth) - optional for mobile
    color_check_passed = True
    if len(img.shape) == 3:  # Color image
        hsv = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv.astype(np.float32))

        if color_variance < 100:  # Relaxed from 200 to 100, but not critical
            color_check_passed = False

    # 5. Calculate final confidence score
    confidence = min(100, int(face_ratio * 3000))  # Adjusted scaling
    confidence = min(confidence, int(laplacian_var * 2))  # Adjusted multiplier
    confidence = min(confidence, int(face_variance))  # Adjusted multiplier

    # Bonus for color variance if available
    if color_check_passed and len(img.shape) == 3:
        confidence = min(100, confidence + 10)

    # Only consider live if confidence is reasonable (relaxed threshold)
    is_live = confidence >= 15  # Relaxed from 30 to 15

    return {
        'is_liveness': is_live,
        'confidence': confidence if is_live else 0,
        'raw_score': confidence / 100.0
    }


def preprocess_for_ocr(image_path, save_path):
    """Preprocess image for better OCR performance."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Error in reading image for OCR preprocessing')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # تحسين حدة الصورة
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(save_path, sharpened)
    print(f"[DEBUG] Preprocessed image saved to {save_path}")
    return save_path


def run_ocr(image_path):
    reader = get_ocr_reader()
    return reader.readtext(image_path, detail=1, paragraph=False)


def extract_student_id_with_confidence(ocr_results):
    """Extract 9-digit student ID from OCR results with flexible matching."""
    print(f"[DEBUG OCR] Processing {len(ocr_results)} OCR results...")
    
    # تخزين جميع الأرقام المكتشفة لأغراض التصحيح
    all_numbers = []
    
    for bbox, text, conf in ocr_results:
        cleaned_text = str(text).strip()
        digits_only = re.sub(r"\D", "", cleaned_text)
        
        print(f"[DEBUG OCR] Text: '{cleaned_text}' → Digits: '{digits_only}' (conf: {conf:.2f})")
        
        if digits_only:
            all_numbers.append((digits_only, conf))
        
        # حاول بالضبط 9 أرقام أولاً
        if len(digits_only) == 9:
            print(f"[DEBUG OCR] ✓ Found exact match: {digits_only}")
            return digits_only, conf
    
    # إذا لم تجد 9 أرقام، حاول 8-10 أرقام (قد يكون هناك خطأ في التعرف)
    for digits, conf in all_numbers:
        if 7 <= len(digits) <= 11:
            print(f"[DEBUG OCR] ✓ Found flexible match: {digits} (length: {len(digits)})")
            return digits, conf
    
    print(f"[DEBUG OCR] ✗ No valid student ID found")
    print(f"[DEBUG OCR] All numbers found: {all_numbers}")
    return None, None


def detect_and_crop_bright_region(image_path, save_path='cropped_card.jpg', debug_path='debug_bright_mask.jpg'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Error in reading')

    original = image.copy()
    h, w = image.shape[:2]

    # 1) convert it to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) blur minimizing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) threshold finding light spots
    _, thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)

    # 4) cleaning
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Mask
    cv2.imwrite(debug_path, thresh)

    # 5) contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imwrite(save_path, original)
        return save_path, False

    # 6) lighter spot in image
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # fail if spot too small
    if area < 0.05 * (h * w):
        cv2.imwrite(save_path, original)
        return save_path, False

    # 7)  bounding rect
    x, y, bw, bh = cv2.boundingRect(largest)

    # 8) padding
    pad_x = int(0.03 * bw)
    pad_y = int(0.03 * bh)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = original[y1:y2, x1:x2]

    cv2.imwrite(save_path, cropped)
    return save_path, True


def extract_student_id_from_card(card_path, timestamp):
    """Extract student ID from ID card image."""
    cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}cropped_card.jpg")
    debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}debug_bright_mask.jpg")
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}ocr_enhanced.jpg")

    print(f"[DEBUG] Starting ID extraction from {card_path}")
    
    cropped_path, found_card = detect_and_crop_bright_region(card_path, cropped_path, debug_path)
    if not found_card:
        print(f"[DEBUG] Warning: Could not find bright region. Using original image.")
        # استخدم الصورة الأصلية إذا فشل اكتشاف المنطقة الساطعة
        cropped_path = card_path

    print(f"[DEBUG] Running OCR on {cropped_path}...")
    processed_path = preprocess_for_ocr(cropped_path, processed_path)
    ocr_results = run_ocr(processed_path)
    
    student_id, conf = extract_student_id_with_confidence(ocr_results)
    print(f"[DEBUG] ID Extraction Result: ID={student_id}, Confidence={conf}")
    
    return student_id, conf


def match_selfie_with_student_record(selfie_path, student_id, timestamp):
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(f'Database not found: {DATABASE_PATH}')

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()

    if student is None:
        return None, None, False

    student_name = student[1] if len(student) > 1 else 'Unknown'
    image_blob = student[6] if len(student) > 6 else None

    if image_blob is None:
        return student_name, None, False

    db_face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}db_face.jpg")
    with open(db_face_path, 'wb') as f:
        f.write(image_blob)

    img1 = cv2.imread(selfie_path)
    img2 = cv2.imread(db_face_path)

    if img1 is None or img2 is None:
        return student_name, None, False

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    analyzer = get_face_analyzer()
    faces1 = analyzer.get(img1)
    faces2 = analyzer.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return student_name, None, False

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return student_name, float(similarity), similarity > SIMILARITY_THRESHOLD


def verify_identity(card_path, selfie_path):
    liveness_result = predict_liveness(selfie_path)
    if not liveness_result['is_liveness']:
        return {
            'success': False,
            'message': 'فشل فحص حيوية الوجه',
            'confidence': liveness_result['confidence']
        }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    student_id, ocr_conf = extract_student_id_from_card(card_path, timestamp)

    if student_id is None:
        return {
            'success': False,
            'message': 'لم يتم العثور على رقم الطالب في البطاقة',
            'confidence': 0
        }

    student_name, similarity, match = match_selfie_with_student_record(selfie_path, student_id, timestamp)
    if not match:
        return {
            'success': False,
            'message': 'الوجه لا يطابق السجل في قاعدة البيانات',
            'confidence': int(similarity * 100) if similarity is not None else 0
        }

    confidence = int((liveness_result['confidence'] + (similarity * 100)) / 2)
    return {
        'success': True,
        'message': f'تم التحقق من الهوية بنجاح - {student_name}',
        'confidence': confidence,
        'student_id': student_id,
        'student_name': student_name,
        'similarity': int(similarity * 100)
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        # التحقق من وجود الملفات
        if 'id_img' not in request.files or 'face_img' not in request.files:
            error = "⚠️ يجب رفع كلا الصورتين"
        else:
            id_img = request.files['id_img']
            face_img = request.files['face_img']

            # التحقق من أن الملفات ليست فارغة
            if id_img.filename == '' or face_img.filename == '':
                error = "⚠️ يجب اختيار الملفات أولاً"
            # التحقق من صيغة الملفات
            elif not (allowed_file(id_img.filename) and allowed_file(face_img.filename)):
                error = "⚠️ الصيغ المسموحة: PNG, JPG, JPEG, GIF فقط"
            else:
                try:
                    # حفظ الملفات برأس آمن
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                    
                    id_filename = secure_filename(timestamp + "id_" + id_img.filename)
                    face_filename = secure_filename(timestamp + "face_" + face_img.filename)
                    
                    id_path = os.path.join(app.config['UPLOAD_FOLDER'], id_filename)
                    face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                    
                    id_img.save(id_path)
                    face_img.save(face_path)

                    # التحقق من الهوية
                    verify_result = verify_identity(id_path, face_path)
                    
                    if verify_result['success']:
                        result = {
                            'type': 'success',
                            'message': verify_result['message'],
                            'confidence': verify_result['confidence'],
                            'id_file': id_filename,
                            'face_file': face_filename
                        }
                    else:
                        result = {
                            'type': 'error',
                            'message': verify_result.get('message', '❌ فشل التحقق')
                        }
                        
                except Exception as e:
                    error = f"❌ خطأ: {str(e)}"

    return render_template("index.html", result=result, error=error)

@app.route("/verify", methods=["POST"])
def verify():
    """
    Route to verify identity from webcam base64 images
    Expects JSON: {
        "id_image": "data:image/jpeg;base64,...",
        "face_image": "data:image/jpeg;base64,..."
    }
    """
    try:
        # استقبال البيانات JSON
        data = request.get_json()
        
        if not data or 'id_image' not in data or 'face_image' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing image data'
            }), 400
        
        # فك تشفير base64 وحفظ الصور مؤقتاً
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        
        try:
            # فك تشفير صورة الهوية
            id_base64 = data['id_image'].split(',')[1] if ',' in data['id_image'] else data['id_image']
            id_image_bytes = base64.b64decode(id_base64)
            id_filename = secure_filename(f"{timestamp}id_capture.jpg")
            id_path = os.path.join(app.config['UPLOAD_FOLDER'], id_filename)
            
            with open(id_path, 'wb') as f:
                f.write(id_image_bytes)
            
            # فك تشفير صورة الوجه
            face_base64 = data['face_image'].split(',')[1] if ',' in data['face_image'] else data['face_image']
            face_image_bytes = base64.b64decode(face_base64)
            face_filename = secure_filename(f"{timestamp}face_capture.jpg")
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            
            with open(face_path, 'wb') as f:
                f.write(face_image_bytes)
        
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error processing images: {str(e)}'
            }), 400
        
        # تشغيل دوال التحقق
        try:
            # التحقق من حيوية الوجه
            liveness_result = predict_liveness(face_path)
            
            if not liveness_result['is_liveness']:
                return jsonify({
                    'success': False,
                    'result': 'ACCESS DENIED',
                    'message': 'فشل فحص حيوية الوجه',
                    'confidence': 0
                })
            
            # استخراج رقم الطالب من البطاقة
            student_id, ocr_conf = extract_student_id_from_card(id_path, timestamp)
            
            if student_id is None:
                return jsonify({
                    'success': False,
                    'result': 'ACCESS DENIED',
                    'message': 'لم يتم العثور على رقم الطالب في البطاقة',
                    'confidence': 0
                })
            
            # مطابقة الوجه مع سجل الطالب في قاعدة البيانات
            student_name, similarity, match = match_selfie_with_student_record(face_path, student_id, timestamp)
            
            if not match:
                return jsonify({
                    'success': False,
                    'result': 'ACCESS DENIED',
                    'message': 'الوجه لا يطابق السجل في قاعدة البيانات',
                    'confidence': int(similarity * 100) if similarity is not None else 0
                })
            
            # نجح التحقق
            confidence = int((liveness_result['confidence'] + (similarity * 100)) / 2)
            
            return jsonify({
                'success': True,
                'result': 'ACCESS GRANTED',
                'message': f'تم التحقق من الهوية بنجاح - {student_name}',
                'confidence': confidence,
                'student_id': student_id,
                'student_name': student_name,
                'similarity': int(similarity * 100),
                'id_file': id_filename,
                'face_file': face_filename
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Verification error: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("index.html", error="❌ الملف كبير جداً (الحد الأقصى 5MB)"), 413

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
