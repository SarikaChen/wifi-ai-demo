# Export a MobileNetV2 Keras model to a TFLite file with basic quantization.
# Usage:
#   python tools/export_mobilenet.py --out ai_sdk_demo/model/model.tflite --img_size 224

import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def export_mobilenet_tflite(out_path: str, img_size: int = 224):
    print(f"Downloading MobileNetV2 (imagenet, {img_size}x{img_size}) ...")
    model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(img_size, img_size, 3))

    print("Converting to TFLite (Optimize.DEFAULT for dynamic-range quantization) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ai_sdk_demo/model/model.tflite", type=str, help="Output .tflite path")
    parser.add_argument("--img_size", default=224, type=int, help="Input size (MobileNetV2 expects 224)")
    args = parser.parse_args()
    export_mobilenet_tflite(args.out, args.img_size)

if __name__ == "__main__":
    main()