import cv2
from doclayout_yolo import YOLOv10
import easyocr
import os
import glob
import fitz  # PyMuPDF
import numpy as np
import json
import csv

# ----------------------------------------------------------------
# ▼▼▼ 使用者変更箇所 ▼▼▼
# ----------------------------------------------------------------
# 1. PDFファイルが格納されているディレクトリのパス
INPUT_DIR = "../検証用pdfデータ"

# 2. 全ての出力（画像、JSON）を保存する親ディレクトリ
OUTPUT_DIR = "output_files_V2" # 保存先フォルダ名を変更

# 3. PDFを画像に変換する際の解像度(dpi)
PDF_DPI = 300

# 4. EasyOCRで認識する言語
LANGUAGES = ['ja', 'en']

# 5. YOLOv10モデルファイルのパス (ローカル)
YOLO_MODEL_PATH = "../doclayout_yolo_docstructbench_imgsz1024.pt"

# 6. YOLO領域拡張量 (ピクセル)
YOLO_EXPAND_PIXELS = 30

# 7. YOLO重複領域とみなす重なり率のしきい値
OVERLAP_THRESHOLD = 0.9
# ----------------------------------------------------------------


def filter_overlapping_boxes(boxes, threshold):
    """
    信頼度に基づき、重複するバウンディングボックスをフィルタリングする。
    重なり率がしきい値以上のボックスのうち、信頼度が低い方を削除する。
    """
    xyxys = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    sorted_indices = np.argsort(confs)[::-1]
    suppressed_indices = []
    for i in range(len(sorted_indices)):
        idx_i = sorted_indices[i]
        if idx_i in suppressed_indices:
            continue
        for j in range(i + 1, len(sorted_indices)):
            idx_j = sorted_indices[j]
            if idx_j in suppressed_indices:
                continue
            box_i, box_j = xyxys[idx_i], xyxys[idx_j]
            inter_xmin, inter_ymin = max(box_i[0], box_j[0]), max(box_i[1], box_j[1])
            inter_xmax, inter_ymax = min(box_i[2], box_j[2]), min(box_i[3], box_j[3])
            inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            smaller_area = min(area_i, area_j)
            if smaller_area == 0:
                continue
            if (inter_area / smaller_area) > threshold:
                suppressed_indices.append(idx_j)
    final_indices_to_keep = [idx for idx in range(len(boxes)) if idx not in suppressed_indices]
    return boxes[final_indices_to_keep]

def process_page_with_easyocr(reader_ocr, img_cv2, filtered_yolo_boxes, yolo_res, pdf_output_dir, page_image_name, img_h, img_w):
    """EasyOCRを使用してページを詳細に解析し、結果を保存する"""
    ocr_results = reader_ocr.readtext(img_cv2, batch_size=4)
    yolo_data_for_json = []
    assigned_ocr_indices = set()
    
    boxes_to_process = [box for box in filtered_yolo_boxes if yolo_res.names[int(box.cls)] != 'abandon']

    for box in boxes_to_process:
        class_id, conf, coords = int(box.cls), float(box.conf), [int(c) for c in box.xyxy[0]]
        label = yolo_res.names[class_id]
        xmin, ymin, xmax, ymax = coords
        expanded_xmin = max(0, xmin - YOLO_EXPAND_PIXELS)
        expanded_ymin = max(0, ymin - YOLO_EXPAND_PIXELS)
        expanded_xmax = min(img_w, xmax + YOLO_EXPAND_PIXELS)
        expanded_ymax = min(img_h, ymax + YOLO_EXPAND_PIXELS)
        
        texts_in_this_box = []
        for i, (ocr_bbox, ocr_text, ocr_prob) in enumerate(ocr_results):
            if i in assigned_ocr_indices: continue
            points = np.array(ocr_bbox, dtype=int)
            tx_min, ty_min = np.min(points, axis=0)
            tx_max, ty_max = np.max(points, axis=0)
            if (expanded_xmin <= tx_min and expanded_ymin <= ty_min and expanded_xmax >= tx_max and expanded_ymax >= ty_max):
                texts_in_this_box.append(ocr_text)
                assigned_ocr_indices.add(i)
        
        yolo_data_for_json.append({
            "label": label, "confidence": round(conf, 4), "box_xyxy": coords,
            "box_leftup": coords[0] + coords[1], "box_area": (coords[2] - coords[0]) * (coords[3] - coords[1]),
            "easyocr_text": " ".join(texts_in_this_box)
        })

    for i, (ocr_bbox, ocr_text, ocr_prob) in enumerate(ocr_results):
        if i not in assigned_ocr_indices:
            points = np.array(ocr_bbox, dtype=int)
            box_xmin, box_ymin = np.min(points, axis=0)
            box_xmax, box_ymax = np.max(points, axis=0)
            yolo_data_for_json.append({
                "label": "nonYOLO", "confidence": 0, "box_xyxy": [int(b) for b in [box_xmin, box_ymin, box_xmax, box_ymax]],
                "box_leftup": int(box_xmin + box_ymin), "box_area": int((box_xmax - box_xmin) * (box_ymax - box_ymin)),
                "easyocr_text": ocr_text
            })
    
    sorted_yolo_data = sorted(yolo_data_for_json, key=lambda x: x['box_leftup'])
    final_processed_data = []
    label_counters = {}
    for item in sorted_yolo_data:
        original_label = item['label']
        current_count = label_counters.get(original_label, 0) + 1
        label_counters[original_label] = current_count
        new_item = item.copy()
        new_item['label'] = f"{original_label}_{current_count}"
        final_processed_data.append(new_item)

    json_save_path = os.path.join(pdf_output_dir, f"{page_image_name}_result.json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(final_processed_data, f, indent=4, ensure_ascii=False)
    
    annotated_image = img_cv2.copy()
    for box in boxes_to_process:
        label, coords = yolo_res.names[int(box.cls)], [int(c) for c in box.xyxy[0]]
        cv2.rectangle(annotated_image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 3)
        cv2.putText(annotated_image, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    for (bbox, text, prob) in ocr_results:
        (top_left, _, bottom_right, _) = bbox
        cv2.rectangle(annotated_image, tuple(map(int, top_left)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
    
    annotated_image_path = os.path.join(pdf_output_dir, f"{page_image_name}_annotated.jpg")
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"    ✅ 結果画像とJSONを保存しました -> {pdf_output_dir}")


def main():
    """
    メイン関数：モデルをロードし、入力ディレクトリからPDFを検索して各ページを処理する。
    """
    print("⏳ モデルをロードしています... (初回は時間がかかる場合があります)")
    try:
        model_yolo = YOLOv10(YOLO_MODEL_PATH)
        reader_ocr = easyocr.Reader(LANGUAGES, gpu=True)
        print("✅ モデルのロードが完了しました。")
    except Exception as e:
        print(f"❌ モデルのロード中にエラーが発生しました: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_data = {}
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(INPUT_DIR, "*.PDF")))
    if not pdf_files:
        print(f"⚠️ エラー: '{INPUT_DIR}' ディレクトリにPDFファイルが見つかりません。")
        return
    print(f"📄 {len(pdf_files)} 件のPDFファイルを検出しました。処理を開始します...")

    for pdf_path in pdf_files:
        pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n====================\n▶️  処理中: {pdf_filename_base}.pdf\n====================")
        pdf_output_dir = os.path.join(OUTPUT_DIR, pdf_filename_base)
        os.makedirs(pdf_output_dir, exist_ok=True)
        summary_data[pdf_filename_base] = {'pymupdf_pages': 0, 'easyocr_pages': 0}

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page_image_name = f"{pdf_filename_base}_page_{page_num + 1}"
                print(f"  - ページ {page_num + 1}/{len(doc)} ({page_image_name}) を処理中...")
                
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=PDF_DPI)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_cv2 = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR if img_data.shape[2] == 4 else cv2.COLOR_RGB2BGR)

                yolo_res = model_yolo.predict(img_cv2, imgsz=1024, conf=0.4, device="mps")[0]
                filtered_yolo_boxes = filter_overlapping_boxes(yolo_res.boxes, OVERLAP_THRESHOLD)

                detected_labels = {yolo_res.names[int(box.cls)] for box in filtered_yolo_boxes}
                print(f"    🔍 検出されたラベル一覧: {detected_labels if detected_labels else 'なし'}")
                
                should_use_pymupdf = detected_labels.issubset({'plain text', 'abandon'})
                img_h, img_w, _ = img_cv2.shape

                if should_use_pymupdf:
                    print("    -> 検出ラベルが単純なため、まずPyMuPDFでの抽出を試みます。")
                    
                    if page.get_text("text").strip():
                        print("    -> PyMuPDFでテキストを構造化します。")
                        summary_data[pdf_filename_base]['pymupdf_pages'] += 1
                        
                        scale_factor = PDF_DPI / 72.0
                        page_dict = page.get_text("dict")
                        pymupdf_spans = []
                        for block in page_dict.get("blocks", []):
                            if block['type'] == 0:
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        scaled_bbox = [
                                            span["bbox"][0] * scale_factor,
                                            span["bbox"][1] * scale_factor,
                                            span["bbox"][2] * scale_factor,
                                            span["bbox"][3] * scale_factor,
                                        ]
                                        pymupdf_spans.append({"bbox": scaled_bbox, "text": span["text"]})
                        
                        yolo_data_for_json = []
                        assigned_span_indices = set()
                        boxes_to_process = [box for box in filtered_yolo_boxes if yolo_res.names[int(box.cls)] != 'abandon']

                        for box in boxes_to_process:
                            class_id, conf, coords = int(box.cls), float(box.conf), [int(c) for c in box.xyxy[0]]
                            label = yolo_res.names[class_id]
                            xmin, ymin, xmax, ymax = coords
                            
                            texts_in_this_box = []
                            for i, span in enumerate(pymupdf_spans):
                                if i in assigned_span_indices: continue
                                sx1, sy1, sx2, sy2 = span["bbox"]
                                if (xmin <= sx1 and ymin <= sy1 and xmax >= sx2 and ymax >= sy2):
                                    texts_in_this_box.append(span["text"])
                                    assigned_span_indices.add(i)
                            
                            yolo_data_for_json.append({
                                "label": label, "confidence": round(conf, 4), "box_xyxy": coords,
                                "box_leftup": coords[0] + coords[1], "box_area": (coords[2] - coords[0]) * (coords[3] - coords[1]),
                                "easyocr_text": " ".join(texts_in_this_box)
                            })
                        
                        for i, span in enumerate(pymupdf_spans):
                            if i not in assigned_span_indices:
                                sx1, sy1, sx2, sy2 = span["bbox"]
                                yolo_data_for_json.append({
                                    "label": "nonYOLO", "confidence": 0, "box_xyxy": [int(b) for b in [sx1, sy1, sx2, sy2]],
                                    "box_leftup": int(sx1 + sy1), "box_area": int((sx2 - sx1) * (sy2 - sy1)),
                                    "easyocr_text": span["text"]
                                })
                        
                        sorted_yolo_data = sorted(yolo_data_for_json, key=lambda x: x['box_leftup'])
                        final_processed_data = []
                        label_counters = {}
                        for item in sorted_yolo_data:
                            original_label, current_count = item['label'], label_counters.get(item['label'], 0) + 1
                            label_counters[original_label] = current_count
                            new_item = item.copy()
                            new_item['label'] = f"{original_label}_{current_count}"
                            final_processed_data.append(new_item)

                        json_save_path = os.path.join(pdf_output_dir, f"{page_image_name}_result.json")
                        with open(json_save_path, "w", encoding="utf-8") as f:
                            json.dump(final_processed_data, f, indent=4, ensure_ascii=False)
                        
                        annotated_image = img_cv2.copy()
                        
                        # ▼▼▼【ここから変更】▼▼▼
                        # 1. PyMuPDFで抽出したスパンのBBoxをオレンジ色で描画
                        for span in pymupdf_spans:
                            sx1, sy1, sx2, sy2 = span["bbox"]
                            cv2.rectangle(annotated_image, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 165, 255), 1) # BGRでオレンジ色

                        # 2. YOLOで検出した領域のBBoxを青色で描画 (既存の処理)
                        for box in boxes_to_process:
                            label, coords = yolo_res.names[int(box.cls)], [int(c) for c in box.xyxy[0]]
                            cv2.rectangle(annotated_image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 3) # BGRで青色
                            cv2.putText(annotated_image, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                        # ▲▲▲【ここまで変更】▲▲▲
                        
                        annotated_image_path = os.path.join(pdf_output_dir, f"{page_image_name}_annotated.jpg")
                        cv2.imwrite(annotated_image_path, annotated_image)
                        print(f"    ✅ 結果画像とJSONを保存しました -> {pdf_output_dir}")

                    else:
                        print("    -> PyMuPDFではテキストを抽出できませんでした。EasyOCRにフォールバックします。")
                        summary_data[pdf_filename_base]['easyocr_pages'] += 1
                        process_page_with_easyocr(reader_ocr, img_cv2, filtered_yolo_boxes, yolo_res, pdf_output_dir, page_image_name, img_h, img_w)
                else:
                    offending_labels = detected_labels - {'plain text', 'abandon'}
                    print(f"    -> '{list(offending_labels)[0]}' など複雑なラベルが検出されたため、EasyOCRで詳細な解析を行います。")
                    summary_data[pdf_filename_base]['easyocr_pages'] += 1
                    process_page_with_easyocr(reader_ocr, img_cv2, filtered_yolo_boxes, yolo_res, pdf_output_dir, page_image_name, img_h, img_w)

            doc.close()
        except Exception as e:
            print(f"❗️ {pdf_filename_base}.pdf の処理中にエラーが発生しました: {e}")

    # --- 8. 全ての処理完了後にCSVサマリーを生成 ---
    print("\n\n🎉🎉🎉 全ての処理が完了しました。 🎉🎉🎉")
    csv_path = os.path.join(OUTPUT_DIR, 'processing_summary.csv')
    print(f"\n📊 処理結果のサマリーをCSVファイルに書き出しています... -> {csv_path}")
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['PDF_Filename', 'PyMuPDF_Pages', 'EasyOCR_Pages', 'Total_Pages'])
            for filename, counts in summary_data.items():
                total_pages = counts['pymupdf_pages'] + counts['easyocr_pages']
                writer.writerow([filename, counts['pymupdf_pages'], counts['easyocr_pages'], total_pages])
        print("✅ サマリーCSVの作成が完了しました。")
    except Exception as e:
        print(f"❌ サマリーCSVの作成中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()