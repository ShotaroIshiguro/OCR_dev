import fitz  # PyMuPDF
import json
import os
import glob

def process_pdfs_in_directory(
    input_dir: str, 
    json_output_dir: str, 
    pdf_output_dir: str,
    draw_bbox: bool = True
):
    """
    指定されたディレクトリ内のすべてのPDFファイルを処理し、テキストと位置情報を抽出し、
    バウンディングボックスを描画した新しいPDFを生成します。

    Args:
        input_dir (str): 入力PDFファイルが格納されているディレクトリ。
        json_output_dir (str): 抽出された情報を保存するJSONファイルの出力先ディレクトリ。
        pdf_output_dir (str): バウンディングボックスを描画したPDFの出力先ディレクトリ。
        draw_bbox (bool): バウンディングボックスを描画するかどうかのフラグ。
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(pdf_output_dir, exist_ok=True)

    # 入力ディレクトリ内の全PDFファイルのパスを取得
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))

    if not pdf_files:
        print(f"エラー: ディレクトリ '{input_dir}' にPDFファイルが見つかりません。")
        return

    print(f"{len(pdf_files)}個のPDFファイルを処理します...")

    # 各PDFファイルをループ処理
    for pdf_path in pdf_files:
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n--- 処理中: {base_filename}.pdf ---")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"エラー: {base_filename}.pdf を開けませんでした: {e}")
            continue

        all_pages_data = []

        # 各ページをループ処理
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # 辞書形式で詳細なテキスト情報を取得
            page_dict = page.get_text("dict")

            page_data = {
                "page_number": page_num + 1,
                "width": page_dict["width"],
                "height": page_dict["height"],
                "blocks": []
            }

            # 抽出した情報をJSON用に整形
            for block in page_dict["blocks"]:
                if block['type'] == 0:  # テキストブロックのみ
                    block_data = { "bbox": block["bbox"], "lines": [] }
                    for line in block["lines"]:
                        line_data = { "bbox": line["bbox"], "spans": [] }
                        for span in line["spans"]:
                            span_data = {
                                "text": span["text"],
                                "font": span["font"],
                                "size": span["size"],
                                "color": span["color"],
                                "bbox": span["bbox"]
                            }
                            line_data["spans"].append(span_data)
                        block_data["lines"].append(line_data)
                    page_data["blocks"].append(block_data)
            
            all_pages_data.append(page_data)

            # バウンディングボックスの描画処理
            if draw_bbox:
                for block in page_data["blocks"]:
                    # 🟪 ブロック (Block) のBBoxを紫色で描画
                    page.draw_rect(fitz.Rect(block["bbox"]), color=(1, 0, 1), width=1.5)
                    for line in block["lines"]:
                        # 🟦 行 (Line) のBBoxを青色で描画
                        page.draw_rect(fitz.Rect(line["bbox"]), color=(0, 0, 1), width=1)
                        for span in line["spans"]:
                            # 🟩 スパン (Span) のBBoxを緑色で描画
                            page.draw_rect(fitz.Rect(span["bbox"]), color=(0, 1, 0), width=0.5)

        # JSONファイルに書き出し
        json_output_path = os.path.join(json_output_dir, f"{base_filename}.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
        print(f"✅ JSONを保存しました: {json_output_path}")

        # 描画済みの新しいPDFを保存
        if draw_bbox:
            pdf_output_path = os.path.join(pdf_output_dir, f"{base_filename}_visualized.pdf")
            doc.save(pdf_output_path)
            print(f"✅ 可視化PDFを保存しました: {pdf_output_path}")

        doc.close()

    print("\n--- 全ての処理が完了しました。 ---")

# --- ここから実行 ---
if __name__ == '__main__':
    # 各ディレクトリのパスを指定
    INPUT_PDF_DIR = "../検証用pdfデータ"
    OUTPUT_JSON_DIR = "pymupdf_test/output_json"
    OUTPUT_PDF_DIR = "pymupdf_test/output_pdfs_visualized"

    # テスト用のダミーPDFファイルを作成
    if not os.path.exists(INPUT_PDF_DIR):
        os.makedirs(INPUT_PDF_DIR)
        print(f"'{INPUT_PDF_DIR}' ディレクトリを作成しました。")

    if not glob.glob(os.path.join(INPUT_PDF_DIR, "*.pdf")):
         print(f"'{INPUT_PDF_DIR}' 内にPDFがないため、ダミーのPDFを作成します。")
         # ダミーファイル1
         doc1 = fitz.open()
         page1 = doc1.new_page()
         page1.insert_text((50, 72), "これは1つ目のPDFのサンプルです。", fontname="gomono", fontsize=11)
         page1.insert_text((50, 100), "First PDF test.", fontname="helv", fontsize=12, color=(0,0,1))
         page1.insert_text((50, 120), "Second line.", fontname="helv", fontsize=12, color=(1,0,0))
         doc1.save(os.path.join(INPUT_PDF_DIR, "sample1.pdf"))
         doc1.close()
         # ダミーファイル2
         doc2 = fitz.open()
         page2 = doc2.new_page()
         page2.insert_text((50, 72), "これは2つ目のPDFです。別ファイルです。", fontname="gomono", fontsize=11)
         doc2.save(os.path.join(INPUT_PDF_DIR, "sample2.pdf"))
         doc2.close()

    # 関数を実行
    process_pdfs_in_directory(
        input_dir=INPUT_PDF_DIR,
        json_output_dir=OUTPUT_JSON_DIR,
        pdf_output_dir=OUTPUT_PDF_DIR
    )