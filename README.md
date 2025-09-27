# AI for Wi-Fi Applications Demo 🚀

這個專案展示了 **AI SDK 部署** 與 **Wi-Fi sensing 應用** 的實作範例，呼應無線產品中 AI 的可能應用場景。  
包含兩個部分：  
1. **AI SDK 部署 Demo** (影像分類 – TensorFlow Lite)  
2. **Wi-Fi Sensing Demo** (WiAR Dataset – 活動辨識)

---

## 📂 專案結構

```
wifi-ai-demo/
├─ ai_sdk_demo/                # AI SDK 部署 Demo (影像分類)
│  ├─ images/                  # 測試圖片 (test.jpg, dog.jpg ...)
│  ├─ model/                   # 模型與標籤檔 (model.tflite, labels.txt)
│  └─ run_inference.py         # 推論程式碼
│
├─ wifi_sensing_demo/          # Wi-Fi Sensing Demo (CSI 活動辨識)
│  ├─ data/                    # WiAR dataset 或 processed.npz
│  ├─ model/                   # 訓練好的模型 & 輸出檔 (h5, tflite, png)
│  └─ train_eval.py            # 訓練 & 測試程式碼
│
├─ tools/                      # 工具腳本 (匯出模型、下載 labels)
│  ├─ export_mobilenet.py
│  └─ get_imagenet_labels.py
│
├─ demo_output/                # Demo 截圖 (ai-sdk-resultpng ...)
│
├─ requirements.txt            # 相依套件
└─ README.md                   # 專案說明文件
```

```yaml
+---------------------------------------------------------+
|                    wifi-ai-demo 專案                    |
+-------------------------+-------------------------------+
|  AI SDK 部署 Demo       |  Wi-Fi Sensing Demo           |
|  (Image Classification) |  (CSI 活動辨識)                |
+-------------------------+-------------------------------+
|                         |                               |
|  Input: test.jpg        |  Input: WiAR CSI Data /       |
|  Model: model.tflite    |         Synthetic Data        |
|  Labels: labels.txt     |                               |
|                         |                               |
|  Run: run_inference.py  |  Run: train_eval.py           |
|                         |                               |
|  Output:                |  Output:                      |
|   - Top-5 predictions   |   - Trained model (.h5)       |
|   - Inference time      |   - TFLite model (.tflite)    |
|                         |   - Accuracy / Confusion      |
|                         |   - training_curve.png        |
|                         |   - training_curve_loss.png   |
+-------------------------+-------------------------------+
```

---

## 1️⃣ AI SDK 部署 Demo
### ✨ 功能
- 使用 **TensorFlow Lite** 部署 **MobileNetV2** 模型  
- 可在 PC / Raspberry Pi / Android 裝置上推論  
- Demo：輸入圖片 → 輸出分類結果  

### 🔧 環境安裝
```bash
pip install -r requirements.txt
```

### ▶️ 執行
```bash
python ai_sdk_demo/run_inference.py   --model ai_sdk_demo/model/model.tflite   --image ai_sdk_demo/images/test.jpg   --labels ai_sdk_demo/model/labels.txt
```

---
## 2️⃣ Wi-Fi Sensing Demo
### ✨ 功能
- 使用 **WiAR dataset** (Wi-Fi CSI Data)  
- 訓練 CNN/LSTM 模型，分類人體活動 (走路 / 坐下 / 揮手...)  
- Demo：輸入 CSI → 輸出動作分類  

### 🔧 環境安裝
```bash
pip install -r requirements.txt
```

### ▶️ 訓練 & 測試
```bash
# 使用合成資料（快速測試流程）
python wifi_sensing_demo/train_eval.py --use_synthetic 1

# 使用實際 WiAR dataset (processed.npz)
python wifi_sensing_demo/train_eval.py
```
---
## 📊 成果展示

### AI SDK 部署 Demo
- 測試圖片：
  ![AI SDK Inference Demo](ai_sdk_demo/ai_sdk_demo/images/test.png)
- Top-5 分類結果：
  ```
  Inference time: 91.51 ms
  Top 1: Egyptian_cat (0.7276)
  Top 2: tabby (0.1734)
  Top 3: tiger_cat (0.0555)
  Top 4: tiger (0.0063)
  Top 5: conch (0.0037)
  ```

- Demo 圖：  
  ![AI SDK Inference Demo](ai_sdk_demo/demo-output/ai-sdk-result.png)

---

### Wi-Fi Sensing Demo
- 測試準確率：`Test accuracy: 1.00`
- 輸出模型：
  - `wifi_sensing_demo/model/csi_cnn.h5`
  - `wifi_sensing_demo/model/csi_cnn.tflite`
- 訓練曲線圖：  
  ![Wi-Fi Sensing Demo](wifi_sensing_demo/model/training_curve.png)

---

## 💡 專案價值
- **AI SDK 部署** → 展示如何將 AI 模型壓縮、最佳化並部署到邊緣裝置。  
- **Wi-Fi Sensing** → 展示 AI 在無線應用（活動偵測、智慧家居、健康監測）的潛力。  
- **延伸應用**  
  - 智慧頻道選擇 / 干擾預測  
  - Wi-Fi 6/7 QoS 優化  
  - Router / IoT 設備中的 AI 助理  

---

## 📝 未來工作
- 模型壓縮（量化 / 剪枝 / 知識蒸餾）  
- 部署到其他邊緣裝置
- 探索更多 Wi-Fi sensing 應用場景  

---

## 👤 作者
- Name: SarikaChen 
- Email: sarika.chen0723@gmail.com  
- LinkedIn: https://www.linkedin.com/in/ling-wei-chen-542a42268/
- GitHub: https://github.com/SarikaChen
