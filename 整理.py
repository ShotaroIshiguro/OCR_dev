import os
import shutil

# --- ▼▼▼ 使用者変更箇所 ▼▼▼ ---

# 整理したいファイルが格納されているディレクトリのパス
SOURCE_DIRECTORY = "output_files_final_4"

# --- ▲▲▲ 変更ここまで ▲▲▲ ---


def organize_files(target_dir):
    """
    指定されたディレクトリ内のファイルを、ファイル名のプレフィックスに基づいて
    サブディレクトリに整理する関数。
    """
    print(f"📁 整理対象ディレクトリ: '{target_dir}'")

    # 1. 対象ディレクトリが存在するかチェック
    if not os.path.isdir(target_dir):
        print(f"❌ エラー: ディレクトリ '{target_dir}' が見つかりません。")
        return

    # 2. ディレクトリ内の全ファイルを取得
    #    サブディレクトリ自体は除外する
    try:
        files_to_move = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    except FileNotFoundError:
        print(f"❌ エラー: ディレクトリ '{target_dir}' の読み込み中に問題が発生しました。")
        return

    if not files_to_move:
        print("✅ 整理対象のファイルはありません。処理を終了します。")
        return

    print(f"🗂️ {len(files_to_move)} 個のファイルを検出しました。整理を開始します...")
    moved_count = 0

    # 3. 各ファイルをループ処理
    for filename in files_to_move:
        # ファイル名に "_page_" が含まれているかチェック
        if "_page_" in filename:
            # ファイル名からPDFのベース名を取得 (例: "mydoc_page_1.jpg" -> "mydoc")
            pdf_base_name = filename.split("_page_")[0]

            # 4. 移動先のサブディレクトリのパスを決定
            destination_dir = os.path.join(target_dir, pdf_base_name)

            # 5. サブディレクトリが存在しない場合は作成
            os.makedirs(destination_dir, exist_ok=True)

            # 6. ファイルを移動
            source_path = os.path.join(target_dir, filename)
            destination_path = os.path.join(destination_dir, filename)

            try:
                shutil.move(source_path, destination_path)
                print(f"  -> 移動しました: {filename}  =>  '{pdf_base_name}/'")
                moved_count += 1
            except Exception as e:
                print(f"❗️ エラー: '{filename}' の移動中にエラーが発生しました: {e}")
        else:
            print(f"  -- スキップ: {filename} (命名規則に一致しません)")

    print(f"\n🎉 整理が完了しました。{moved_count}個のファイルを移動しました。")


if __name__ == "__main__":
    organize_files(SOURCE_DIRECTORY)