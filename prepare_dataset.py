"""
Автоматическая подготовка датасета для LoRA обучения.
Делает ВСЁ за тебя:
1. Находит все фотки в папке
2. Детектит лицо на каждой (нейросетевой детектор)
3. Создаёт кроп лица (face_xxx.jpg)
4. Создаёт .txt файлы с trigger word
5. Убирает размытые фото

Использование:
    python prepare_dataset.py <папка_с_фотками> <trigger_word>
    python prepare_dataset.py C:\photos\misu misu
"""

import cv2
import numpy as np
import os
import sys
import shutil
import urllib.request
from pathlib import Path

# Пути к DNN модели
DNN_PROTO = None
DNN_MODEL = None
DNN_NET = None

def ensure_dnn_model():
    """Скачивает DNN модель для точного детектирования лица"""
    global DNN_PROTO, DNN_MODEL, DNN_NET
    
    model_dir = Path(__file__).parent / "face_model"
    model_dir.mkdir(exist_ok=True)
    
    proto_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    if not proto_path.exists():
        print("  Скачиваю DNN модель детектора лица (один раз)...")
        urllib.request.urlretrieve(proto_url, str(proto_path))
    
    if not model_path.exists():
        print("  Скачиваю веса модели (~10MB)...")
        urllib.request.urlretrieve(model_url, str(model_path))
    
    DNN_PROTO = str(proto_path)
    DNN_MODEL = str(model_path)
    DNN_NET = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    print("  ✅ DNN детектор лица загружен\n")


def detect_face_dnn(img, confidence_threshold=0.3):
    """Детектит лицо через DNN (SSD ResNet) — НАМНОГО точнее Haar cascade"""
    global DNN_NET
    
    h, w = img.shape[:2]
    
    # Пробуем оригинал + много углов поворота (для наклонённых selfie)
    angles = [0, 90, -90, 180, 45, -45, 135, -135]
    best_face = None
    best_conf = 0
    
    for angle in angles:
        if angle == 0:
            test_img = img
            th, tw = h, w
        else:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            tw = int(h * sin + w * cos)
            th = int(h * cos + w * sin)
            M[0, 2] += (tw - w) / 2
            M[1, 2] += (th - h) / 2
            test_img = cv2.warpAffine(img, M, (tw, th))
        
        blob = cv2.dnn.blobFromImage(test_img, 1.0, (300, 300),
                                      [104, 117, 123], False, False)
        DNN_NET.setInput(blob)
        detections = DNN_NET.forward()
        
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > confidence_threshold and conf > best_conf:
                x1 = int(detections[0, 0, i, 3] * tw)
                y1 = int(detections[0, 0, i, 4] * th)
                x2 = int(detections[0, 0, i, 5] * tw)
                y2 = int(detections[0, 0, i, 6] * th)
                
                if angle == 0:
                    best_face = (x1, y1, x2 - x1, y2 - y1)
                    best_conf = conf
                else:
                    # Для повёрнутых — обратная трансформация координат
                    cx_rot = (x1 + x2) / 2
                    cy_rot = (y1 + y2) / 2
                    fw_rot = x2 - x1
                    fh_rot = y2 - y1
                    
                    M_inv = cv2.invertAffineTransform(M)
                    pt = np.array([cx_rot, cy_rot, 1.0])
                    cx_orig = M_inv[0] @ pt
                    cy_orig = M_inv[1] @ pt
                    
                    # Размер лица при повороте может переключать w/h
                    if angle in [90, -90]:
                        fw_orig, fh_orig = fh_rot, fw_rot
                    else:
                        fw_orig, fh_orig = fw_rot, fh_rot
                    
                    ox1 = max(0, int(cx_orig - fw_orig / 2))
                    oy1 = max(0, int(cy_orig - fh_orig / 2))
                    
                    best_face = (ox1, oy1, int(fw_orig), int(fh_orig))
                    best_conf = conf
    
    return best_face


def is_blurry(img, threshold=25):
    """Проверяет размытость через Лапласиан"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def crop_face(img, face_bbox, padding=0.6):
    """Кропает лицо с паддингом для плечей, квадратный формат"""
    h, w = img.shape[:2]
    x, y, fw, fh = face_bbox
    
    # Паддинг
    pad_x = int(fw * padding)
    pad_y_top = int(fh * padding * 0.5)    # меньше сверху
    pad_y_bottom = int(fh * padding * 1.5)  # больше снизу (плечи)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y_top)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y_bottom)
    
    # Квадрат
    crop_w = x2 - x1
    crop_h = y2 - y1
    size = max(crop_w, crop_h)
    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    
    # Если квадрат не влезает — корректируем
    if x2 - x1 != y2 - y1:
        size = min(x2 - x1, y2 - y1)
        x2 = x1 + size
        y2 = y1 + size
    
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        crop = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    return crop


def prepare_dataset(input_dir, trigger_word, output_dir=None):
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.name}_prepared"
    output_path = Path(output_dir)
    
    full_dir = output_path / "full"
    face_dir = output_path / "face"
    full_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    print(f"\n{'='*50}")
    print(f"  Подготовка датасета для: {trigger_word}")
    print(f"  Найдено фоток: {len(images)}")
    print(f"  Выход: {output_path}")
    print(f"{'='*50}\n")
    
    stats = {'total': len(images), 'blurry': 0, 'no_face': 0, 'ok': 0}
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}...", end=" ")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("❌ не удалось загрузить")
            continue
        
        # Проверка размытости
        if is_blurry(img):
            print("⚠️  размытое — пропускаю")
            stats['blurry'] += 1
            continue
        
        # --- Сохраняем полное фото ---
        out_full = full_dir / f"{img_path.stem}.jpg"
        # Ресайз если слишком большое (макс сторона 1536)
        h, w = img.shape[:2]
        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), 
                           interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_full), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Txt для полного фото
        txt_full = full_dir / f"{img_path.stem}.txt"
        txt_full.write_text(trigger_word, encoding='utf-8')
        
        # --- Кроп лица ---
        face = detect_face_dnn(img)
        if face is not None:
            face_crop = crop_face(img, face)
            if face_crop.shape[0] > 0:
                out_face = face_dir / f"face_{img_path.stem}.jpg"
                cv2.imwrite(str(out_face), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                txt_face = face_dir / f"face_{img_path.stem}.txt"
                txt_face.write_text(trigger_word, encoding='utf-8')
                
                print("✅ фото + кроп лица")
                stats['ok'] += 1
            else:
                print("✅ фото (кроп неудачный)")
                stats['ok'] += 1
        else:
            print("✅ фото (лицо не найдено — кроп пропущен)")
            stats['no_face'] += 1
            stats['ok'] += 1
    
    # Итог
    face_count = len(list(face_dir.glob("*.jpg")))
    full_count = len(list(full_dir.glob("*.jpg")))
    
    print(f"\n{'='*50}")
    print(f"  ГОТОВО!")
    print(f"  Полных фото:  {full_count}  ({full_dir})")
    print(f"  Кропов лица:  {face_count}  ({face_dir})")
    print(f"  Пропущено размытых: {stats['blurry']}")
    print(f"  Без детекции лица:  {stats['no_face']}")
    print(f"{'='*50}")
    print(f"\nТеперь в конфиге LoRA укажи ОБЕ папки как datasets:")
    print(f'  - folder_path: "{face_dir}"')
    print(f'  - folder_path: "{full_dir}"')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python prepare_dataset.py <папка> <trigger_word>")
        print("Пример: python prepare_dataset.py C:\\photos\\misu misu")
        sys.exit(1)
    
    ensure_dnn_model()
    prepare_dataset(sys.argv[1], sys.argv[2])
