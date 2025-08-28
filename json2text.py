import os
import json

def aggregate_ocr_texts(root_directory):
    """
    指定されたルートディレクトリ内の各サブディレクトリ（書類）にある
    OCR結果のJSONファイルからテキストを抽出し、
    書類ごとに1つのテキストファイルにまとめる。

    Args:
        root_directory (str): 書類名のディレクトリが格納されている親ディレクトリのパス。
    """
    # 指定されたルートディレクトリが存在するかチェック
    if not os.path.isdir(root_directory):
        print(f"エラー: ディレクトリ '{root_directory}' が見つかりません。")
        return

    print(f"処理を開始します。ルートディレクトリ: {root_directory}")

    # ルートディレクトリ内の各エントリ（サブディレクトリを想定）をループ
    for doc_name in os.listdir(root_directory):
        doc_path = os.path.join(root_directory, doc_name)

        # エントリがディレクトリの場合のみ処理を続行
        if os.path.isdir(doc_path):
            print(f"\n📄 書類 '{doc_name}' を処理中...")
            all_page_texts = []
            
            # ページ順に処理するためにファイル名をソートする
            json_files = sorted([f for f in os.listdir(doc_path) if f.endswith('.json')])

            if not json_files:
                print(f"  - JSONファイルが見つかりませんでした。")
                continue

            # ディレクトリ内の各JSONファイルをループ
            for json_filename in json_files:
                json_filepath = os.path.join(doc_path, json_filename)

                try:
                    with open(json_filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # JSON内の各オブジェクトからテキストを抽出
                        for item in data:
                            if "easyocr_text" in item and item["easyocr_text"]:
                                all_page_texts.append(item["easyocr_text"])

                except json.JSONDecodeError:
                    print(f"  - 警告: '{json_filename}' は不正なJSONファイルです。スキップします。")
                except Exception as e:
                    print(f"  - エラー: '{json_filename}' の処理中にエラーが発生しました: {e}")

            # 抽出したテキストがあれば、テキストファイルに書き出す
            if all_page_texts:
                output_filename = f"{doc_name}.txt"
                output_filepath = os.path.join(root_directory, output_filename) # ルートディレクトリに保存

                with open(output_filepath, 'w', encoding='utf-8') as f:
                    # 各ページのテキストを改行で区切って書き込む
                    f.write("\n".join(all_page_texts))
                
                print(f"  ✅ テキストファイル '{output_filename}' を作成しました。")
            else:
                print(f"  - テキストが抽出されなかったため、ファイルは作成されませんでした。")

    print("\nすべての処理が完了しました。")

# --- 実行 ---
# このスクリプトを実行する前に、以下の `root_directory` を
# あなたの環境に合わせて変更してください。
if __name__ == '__main__':
    # 例: 'C:/Users/YourUser/Documents/OCR_Results' のような形式で指定
    root_directory = './output_files_V1'
    aggregate_ocr_texts(root_directory)