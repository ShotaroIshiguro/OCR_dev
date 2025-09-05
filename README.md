# OCR_dev

## 仕様
- V1採用
- 

## やることリスト
- コンテナ化
- ボックスの位置関係を最適化
- YOLOの性能が低いせいで正常なテキストなのにnonYOLOラベルが振られている
  - ラベル順にnonYOLOの右上座標と左上座標が近ければ(10pixel)まとめる処理を追加（グルーピンぐ）
  - グルーピングはできたっぽいから、ソートの順番を変える
  - ラベル順番の振り直し
  - グルーピングできたらplaintext
- 画像の縦横比によるラベル振り直しを考える　
- 一定の文字数以下のabandonを削除（文字数要検討）


## コマンド
```
docker build -t yolo-easy-pymupdf .
```

```
docker run --rm \
  -v ./テストファイル:/app/テストファイル \
  -v ./output_files_V1:/app/output_files_V1 \
  yolo-easy-pymupdf
```