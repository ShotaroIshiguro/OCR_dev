# OCR_dev

## 仕様
- V1採用
- 

## やることリスト
- コンテナ化
- nonYOLOをabandon
- 一定の文字数以下のabandonを削除


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