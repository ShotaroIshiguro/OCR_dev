import os
import subprocess

def convert_dir_to_pdf(input_dir, output_dir):
    """
    指定されたディレクトリ内のOfficeファイルをすべてPDFに変換します。
    macOS/Linux (LibreOfficeインストール済み) での動作を想定しています。

    :param input_dir: 変換したいファイルが入っているディレクトリのパス
    :param output_dir: PDFを出力するディレクトリのパス
    """
    # 対応する拡張子
    supported_extensions = ('.docx', '.xlsx', '.pptx')

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    print(f"ディレクトリのスキャンを開始: {input_dir}")

    # 入力ディレクトリ内のファイルをループ処理
    for filename in os.listdir(input_dir):
        # ファイルの拡張子が対応しているかチェック
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_dir, filename)
            
            # LibreOfficeのコマンドを実行
            command = [
                'soffice',     # Homebrewでインストールした場合のコマンド
                '--headless',      # GUIなしで実行
                '--convert-to',    # 変換を指定
                'pdf',             # 変換形式
                '--outdir',        # 出力ディレクトリ
                output_dir,
                input_path
            ]
            
            try:
                print(f"変換中: {filename}")
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"変換完了: {filename} -> PDF")
            except FileNotFoundError:
                print("\nエラー: 'libreoffice' コマンドが見つかりません。")
                print("HomebrewでLibreOfficeが正しくインストールされているか確認してください。")
                # より直接的なパスを指定する場合の例
                # full_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
                # print(f"または、'{full_path}' のようなフルパスを試してください。")
                return # エラーが見つかったら処理を中断
            except subprocess.CalledProcessError as e:
                print(f"エラー: {filename} の変換に失敗しました。")
                print(f"エラー内容:\n{e.stderr.decode('utf-8', 'ignore')}")

    print("\n全ての処理が完了しました。")

# --- 使い方 ---
# 変換したいファイルが入っているディレクトリを指定
input_directory = './検証用その他データ'

# PDFの保存先ディレクトリを指定
output_directory = './検証用その他データ'


# 関数を実行
convert_dir_to_pdf(input_directory, output_directory)