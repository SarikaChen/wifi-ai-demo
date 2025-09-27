# tools/get_imagenet_labels.py
# 自動下載 ImageNet 1000 類別標籤，輸出到 ai_sdk_demo/model/labels.txt

import json, urllib.request, os

URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
out_path = "ai_sdk_demo/model/labels.txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with urllib.request.urlopen(URL) as r:
    idx2label = json.load(r)  # {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}

labels = [idx2label[str(i)][1] for i in range(1000)]
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(labels))

print(f"Wrote {len(labels)} labels to {out_path}")
