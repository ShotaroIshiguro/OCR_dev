import cv2
from doclayout_yolo import YOLOv10
import easyocr
import os
import glob
import fitz  # PyMuPDF
import numpy as np
import json
import csv
import boto3 # 追加
import sys   # 追加

# ----------------------------------------------------------------
# ▼▼▼ 設定箇所 ▼▼▼ (一部は環境変数で上書きされます)
# ----------------------------------------------------------------
# 3. PDFを画像に変換する際の解像度(dpi)
PDF_DPI = 300

# 4. EasyOCRで認識する言語
LANGUAGES = ['ja', 'en']

# 5. YOLOv10モデルファイルのパス (コンテナ内のパス)
YOLO_MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"

# 6. YOLO領域拡張量 (ピクセル)
YOLO_EXPAND_PIXELS_X = 50 # 横方向の拡張量
YOLO_EXPAND_PIXELS_Y = 50 # 縦方向の拡張量

# 7. YOLO重複領域とみなす重なり率のしきい値
OVERLAP_THRESHOLD = 0.9

# 8. nonYOLOテキストのグループ化しきい値 (ピクセル)
NON_YOLO_LINE_MERGE_H_THRESHOLD = 100
NON_YOLO_LINE_MERGE_V_THRESHOLD = 50
NON_YOLO_PARA_MERGE_V_THRESHOLD = 50
NON_YOLO_PARA_MIN_X_OVERLAP_RATIO = 0.01

# 9. nonYOLOグループ化前のソート時のY座標許容誤差 (ピクセル)
GROUPING_SORT_Y_TOLERANCE = 30
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# ▼▼▼ これより下の関数は変更なし ▼▼▼
# (filter_overlapping_boxes, group_non_yolo_ocr_results, process_page_with_easyocr)
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

def group_non_yolo_ocr_results(unassigned_ocr_results):
    """
    未割り当てのOCR結果（nonYOLO候補）を、まず単語を行に、次に行を段落に結合する。
    """
    if not unassigned_ocr_results:
        return []

    boxes = []
    for i, (ocr_bbox, ocr_text, ocr_prob) in enumerate(unassigned_ocr_results):
        points = np.array(ocr_bbox)
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        boxes.append({'id': i, 'bbox': [xmin, ymin, xmax, ymax], 'text': ocr_text, 'points': points.tolist(), 'center': (center_x, center_y)})

    adj = {box['id']: [] for box in boxes}
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1 = boxes[i]
            b2 = boxes[j]
            box1, box2 = b1['bbox'], b2['bbox']
            vertical_gap_line = max(0, box2[1] - box1[3], box1[1] - box2[3])
            is_vertically_aligned_for_line = vertical_gap_line < NON_YOLO_LINE_MERGE_V_THRESHOLD
            horizontal_gap_line = max(0, box2[0] - box1[2], box1[0] - box2[2])
            is_horizontally_close_for_line = horizontal_gap_line < NON_YOLO_LINE_MERGE_H_THRESHOLD
            if is_vertically_aligned_for_line and is_horizontally_close_for_line:
                adj[b1['id']].append(b2['id'])
                adj[b2['id']].append(b1['id'])
                continue
            vertical_gap_para = max(0, box2[1] - box1[3], box1[1] - box2[3])
            is_vertically_close_for_para = vertical_gap_para < NON_YOLO_PARA_MERGE_V_THRESHOLD
            overlap_x = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
            min_width = min(box1[2] - box1[0], box2[2] - box2[0])
            is_horizontally_aligned_for_para = (min_width > 0 and (overlap_x / min_width) > NON_YOLO_PARA_MIN_X_OVERLAP_RATIO)
            if is_vertically_close_for_para and is_horizontally_aligned_for_para:
                adj[b1['id']].append(b2['id'])
                adj[b2['id']].append(b1['id'])

    visited = set()
    groups = []
    for box in boxes:
        box_id = box['id']
        if box_id not in visited:
            component_indices = []
            stack = [box_id]
            visited.add(box_id)
            while stack:
                node_idx = stack.pop()
                component_indices.append(node_idx)
                for neighbor_idx in adj[node_idx]:
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        stack.append(neighbor_idx)
            groups.append(component_indices)

    grouped_blocks = []
    id_to_box_map = {box['id']: box for box in boxes}
    for group_indices in groups:
        component_boxes = [id_to_box_map[i] for i in group_indices]
        if not component_boxes:
            continue
        y_sorted_boxes = sorted(component_boxes, key=lambda b: b['center'][1])
        lines = []
        if y_sorted_boxes:
            current_line = [y_sorted_boxes[0]]
            for i in range(1, len(y_sorted_boxes)):
                line_avg_y = sum(b['center'][1] for b in current_line) / len(current_line)
                current_box = y_sorted_boxes[i]
                if abs(current_box['center'][1] - line_avg_y) < GROUPING_SORT_Y_TOLERANCE:
                    current_line.append(current_box)
                else:
                    lines.append(current_line)
                    current_line = [current_box]
            lines.append(current_line)
        sorted_component_boxes = []
        for line in lines:
            sorted_line = sorted(line, key=lambda b: b['center'][0])
            sorted_component_boxes.extend(sorted_line)
        all_points = [p for box in sorted_component_boxes for p in box['points']]
        all_texts = [box['text'] for box in sorted_component_boxes]
        if not all_points:
            continue
        final_points = np.array(all_points)
        final_xmin, final_ymin = np.min(final_points, axis=0)
        final_xmax, final_ymax = np.max(final_points, axis=0)
        final_box_xyxy = [int(final_xmin), int(final_ymin), int(final_xmax), int(final_ymax)]
        final_box_center = (final_xmin + final_xmax) / 2, (final_ymin + final_ymax) / 2
        final_text = " ".join(all_texts).strip()
        grouped_blocks.append({
            "label": "nonYOLO", "confidence": 0, "box_xyxy": final_box_xyxy,
            "box_top_left": (final_box_xyxy[0], final_box_xyxy[1]),
            "box_bottom_right": (final_box_xyxy[2], final_box_xyxy[3]),
            "box_area": int((final_xmax - final_xmin) * (final_ymax - final_ymin)),
            "easyocr_text": final_text, "box_center": final_box_center
        })
    return grouped_blocks

def process_page_with_easyocr(reader_ocr, img_cv2, filtered_yolo_boxes, yolo_res, local_output_dir, page_image_name, img_h, img_w):
    """EasyOCRを使用してページを詳細に解析し、結果を保存する"""
    ocr_results = reader_ocr.readtext(img_cv2, batch_size=4, link_threshold=0.3)
    yolo_data_for_json = []
    assigned_ocr_indices = set()
    boxes_to_process = [box for box in filtered_yolo_boxes if yolo_res.names[int(box.cls)] != 'abandon']

    for box in boxes_to_process:
        class_id, conf, coords = int(box.cls), float(box.conf), [int(c) for c in box.xyxy[0]]
        label = yolo_res.names[class_id]
        xmin, ymin, xmax, ymax = coords
        expanded_xmin, expanded_ymin = max(0, xmin - YOLO_EXPAND_PIXELS_X), max(0, ymin - YOLO_EXPAND_PIXELS_Y)
        expanded_xmax, expanded_ymax = min(img_w, xmax + YOLO_EXPAND_PIXELS_X), min(img_h, ymax + YOLO_EXPAND_PIXELS_Y)
        box_center = (xmin + xmax) / 2, (ymin + ymax) / 2
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
            "box_top_left": (coords[0], coords[1]), "box_bottom_right": (coords[2], coords[3]),
            "box_area": (coords[2] - coords[0]) * (coords[3] - coords[1]), "easyocr_text": " ".join(texts_in_this_box).strip(),
            "box_center": box_center
        })

    unassigned_ocr_results = [(ocr_bbox, ocr_text, ocr_prob) for i, (ocr_bbox, ocr_text, ocr_prob) in enumerate(ocr_results) if i not in assigned_ocr_indices]
    grouped_non_yolo_data = group_non_yolo_ocr_results(unassigned_ocr_results)
    if grouped_non_yolo_data:
        print(f"    🤝 nonYOLOテキストのグループ化結果: {[item['easyocr_text'] for item in grouped_non_yolo_data]}")
    yolo_data_for_json.extend(grouped_non_yolo_data)
    
    if yolo_data_for_json:
        aspect_ratio = (img_h / img_w) if img_w > 0 else 1
        def sort_key(item):
            x_leftup, y_leftup = item['box_top_left']
            return x_leftup + y_leftup * aspect_ratio
        sorted_yolo_data = sorted(yolo_data_for_json, key=sort_key)
    else:
        sorted_yolo_data = []

    final_processed_data = []
    label_counters = {}
    for item in sorted_yolo_data:
        original_label = item['label']
        current_count = label_counters.get(original_label, 0) + 1
        label_counters[original_label] = current_count
        new_item = item.copy()
        new_item['label'] = f"{original_label}_{current_count}"
        final_processed_data.append(new_item)

    json_save_path = os.path.join(local_output_dir, f"{page_image_name}_result.json")
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
    
    annotated_image_path = os.path.join(local_output_dir, f"{page_image_name}_annotated.jpg")
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"    ✅ 結果画像とJSONをローカルに保存しました -> {local_output_dir}")
# ----------------------------------------------------------------
# ▲▲▲ 上記の関数は変更なし ▲▲▲
# ----------------------------------------------------------------


def main():
    """
    メイン関数：環境変数からS3のファイル情報を取得し、単一のPDFを処理して結果をS3にアップロードする。
    """
    # --- 1. 環境変数から情報を取得 ---
    try:
        s3_bucket = os.environ['S3_BUCKET']
        s3_input_key = os.environ['S3_INPUT_KEY']
        s3_output_prefix = os.environ['S3_OUTPUT_PREFIX']
    except KeyError as e:
        print(f"❌ エラー: 必要な環境変数が設定されていません: {e}")
        sys.exit(1)

    pdf_filename_base = os.path.splitext(os.path.basename(s3_input_key))[0]
    
    # --- 2. ローカルの一時作業ディレクトリを設定 ---
    local_pdf_path = f"/tmp/{pdf_filename_base}.pdf"
    local_output_dir = f"/tmp/{pdf_filename_base}_output"
    os.makedirs(local_output_dir, exist_ok=True)
    print(f"⚙️ ローカル作業ディレクトリ: {local_output_dir}")

    # --- 3. S3からPDFファイルをダウンロード ---
    s3 = boto3.client('s3')
    try:
        print(f"⬇️ S3からファイルをダウンロード中: s3://{s3_bucket}/{s3_input_key}")
        s3.download_file(s3_bucket, s3_input_key, local_pdf_path)
        print("✅ ダウンロード完了。")
    except Exception as e:
        print(f"❌ S3からのダウンロード中にエラーが発生しました: {e}")
        sys.exit(1)

    # --- 4. モデルをロード ---
    print("⏳ モデルをロードしています...")
    try:
        model_yolo = YOLOv10(YOLO_MODEL_PATH)
        reader_ocr = easyocr.Reader(LANGUAGES, gpu=False) # ECS(EC2)のCPU環境を想定
        print("✅ モデルのロードが完了しました。")
    except Exception as e:
        print(f"❌ モデルのロード中にエラーが発生しました: {e}")
        sys.exit(1)

    # --- 5. PDF処理の実行 ---
    print(f"\n====================\n▶️  処理中: {pdf_filename_base}.pdf\n====================")
    summary_data = {'pymupdf_pages': 0, 'easyocr_pages': 0}
    
    try:
        doc = fitz.open(local_pdf_path)
        for page_num in range(len(doc)):
            page_image_name = f"{pdf_filename_base}_page_{page_num + 1}"
            print(f"  - ページ {page_num + 1}/{len(doc)} ({page_image_name}) を処理中...")
            
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=PDF_DPI)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_cv2 = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR if img_data.shape[2] == 4 else cv2.COLOR_RGB2BGR)

            yolo_res = model_yolo.predict(img_cv2, imgsz=1024, conf=0.4)[0]
            filtered_yolo_boxes = filter_overlapping_boxes(yolo_res.boxes, OVERLAP_THRESHOLD)
            detected_labels = {yolo_res.names[int(box.cls)] for box in filtered_yolo_boxes}
            print(f"    🔍 検出されたラベル一覧: {detected_labels if detected_labels else 'なし'}")
            
            should_use_pymupdf = detected_labels.issubset({'plain text', 'abandon'})
            img_h, img_w, _ = img_cv2.shape

            if should_use_pymupdf:
                print("    -> 検出ラベルが単純なため、まずPyMuPDFでの抽出を試みます。")
                full_page_text = page.get_text("text").strip()
                if full_page_text:
                    summary_data['pymupdf_pages'] += 1
                    final_processed_data = [{"label": "plain_text_1", "confidence": 1.0, "box_xyxy": [0, 0, img_w, img_h], "easyocr_text": full_page_text}]
                    json_save_path = os.path.join(local_output_dir, f"{page_image_name}_result.json")
                    with open(json_save_path, "w", encoding="utf-8") as f:
                        json.dump(final_processed_data, f, indent=4, ensure_ascii=False)
                    annotated_image = img_cv2.copy()
                    cv2.rectangle(annotated_image, (0, 0), (img_w, img_h), (255, 0, 0), 5)
                    cv2.putText(annotated_image, "Processed by PyMuPDF", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4)
                    annotated_image_path = os.path.join(local_output_dir, f"{page_image_name}_annotated.jpg")
                    cv2.imwrite(annotated_image_path, annotated_image)
                    print(f"    ✅ PyMuPDFの結果をローカルに保存しました。")
                else:
                    print("    -> PyMuPDFではテキストを抽出できませんでした。EasyOCRにフォールバックします。")
                    summary_data['easyocr_pages'] += 1
                    process_page_with_easyocr(reader_ocr, img_cv2, filtered_yolo_boxes, yolo_res, local_output_dir, page_image_name, img_h, img_w)
            else:
                offending_labels = detected_labels - {'plain text', 'abandon'}
                print(f"    -> '{list(offending_labels)[0]}' など複雑なラベルが検出されたため、EasyOCRで詳細な解析を行います。")
                summary_data['easyocr_pages'] += 1
                process_page_with_easyocr(reader_ocr, img_cv2, filtered_yolo_boxes, yolo_res, local_output_dir, page_image_name, img_h, img_w)
        doc.close()
    except Exception as e:
        print(f"❗️ {pdf_filename_base}.pdf の処理中にエラーが発生しました: {e}")
        sys.exit(1)

    # --- 6. 処理サマリーCSVをローカルに作成 ---
    total_pages = summary_data['pymupdf_pages'] + summary_data['easyocr_pages']
    csv_path = os.path.join(local_output_dir, f'{pdf_filename_base}_summary.csv')
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['PDF_Filename', 'PyMuPDF_Pages', 'EasyOCR_Pages', 'Total_Pages'])
            writer.writerow([f"{pdf_filename_base}.pdf", summary_data['pymupdf_pages'], summary_data['easyocr_pages'], total_pages])
        print(f"📊 サマリーCSVをローカルに作成しました -> {csv_path}")
    except Exception as e:
        print(f"❌ サマリーCSVの作成中にエラーが発生しました: {e}")

    # --- 7. 全ての成果物をS3にアップロード ---
    print(f"\n⬆️ 全ての成果物をS3にアップロード中: s3://{s3_bucket}/{s3_output_prefix}/")
    try:
        for root, _, files in os.walk(local_output_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = f"{s3_output_prefix}/{file}"
                s3.upload_file(local_file_path, s3_bucket, s3_key)
                print(f"  -> Uploaded {file}")
        print("✅ 全てのファイルのアップロードが完了しました。")
    except Exception as e:
        print(f"❌ S3へのアップロード中にエラーが発生しました: {e}")
        sys.exit(1)

    print("\n\n🎉🎉🎉 処理が正常に完了しました。 🎉🎉🎉")

if __name__ == "__main__":
    main()