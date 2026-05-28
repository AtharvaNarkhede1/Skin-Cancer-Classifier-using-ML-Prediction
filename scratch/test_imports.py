
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    import google.protobuf
    print(f"Protobuf version: {google.protobuf.__version__}")
    print("TensorFlow import successful!")
except Exception as e:
    print(f"TensorFlow import failed: {e}")

try:
    import google.generativeai as genai
    print("google.generativeai import successful!")
except Exception as e:
    print(f"google.generativeai import failed: {e}")
