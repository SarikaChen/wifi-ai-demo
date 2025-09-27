import argparse
import numpy as np
from PIL import Image
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def load_labels(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f]


def preprocess_image(image_path, input_size):
    img = Image.open(image_path).convert("RGB").resize(input_size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser(description="TFLite image classification demo")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels", default=None, help="Optional labels file (one per line)")
    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, h, w, _ = input_details[0]["shape"]
    input_data = preprocess_image(args.image, (w, h))

    interpreter.set_tensor(input_details[0]["index"], input_data)

    t0 = time.time()
    interpreter.invoke()
    dt = (time.time() - t0) * 1000.0

    output = interpreter.get_tensor(output_details[0]["index"]).squeeze()

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    if output.ndim == 1 and np.all(output >= 0) and abs(output.sum() - 1.0) < 1e-3:
        probs = output
    else:
        probs = softmax(output)

    top_k = probs.argsort()[-5:][::-1]
    labels = load_labels(args.labels)

    print(f"Inference time: {dt:.2f} ms")
    for i, idx in enumerate(top_k):
        label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
        print(f"Top {i+1}: {label} ({probs[idx]:.4f})")


if __name__ == "__main__":
    main()