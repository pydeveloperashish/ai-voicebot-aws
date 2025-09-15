import os
import tarfile
import onnxruntime as ort
import numpy as np

class Predictor:
    def __init__(self, model_path):
        providers = [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CUDAExecutionProvider",
        ]
        # create session (exceptions will bubble up)
        self.session = ort.InferenceSession(model_path, providers=providers)

    def predict(self, input_data):
        # Example: input_data = {"input_ids": [[101, 2009, 2003, 1037, 3944, 102]], "attention_mask": [[1,1,1,1,1,1]]}
        inputs = {k: np.array(v, dtype=np.int64) for k, v in input_data.items()}
        outputs = self.session.run(None, inputs)
        return outputs[0].tolist()

model = None

# ------------------ new model_fn ------------------
def _ensure_extracted(model_dir):
    """If model.tar.gz exists in model_dir, extract it in-place."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        print(f"Found archive {tar_path} — extracting into {model_dir} ...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=model_dir)
            print("Extraction finished.")
        except Exception as e:
            print("Extraction failed:", e)
            raise

def _find_first_onnx(root_dir):
    """Recursively find the first .onnx file under root_dir. Return full path or None."""
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".onnx"):
                return os.path.join(root, f)
    return None

def model_fn(model_dir):
    """
    SageMaker-style model_fn:
    - extract model.tar.gz if present (custom container behavior)
    - find .onnx file recursively and load Predictor using that file
    """
    print("model_fn called. model_dir:", model_dir)
    # 1) extract if needed
    try:
        _ensure_extracted(model_dir)
    except Exception as e:
        print("Error while extracting model archive:", e)
        raise

    # 2) find ONNX
    onnx_path = _find_first_onnx(model_dir)
    if not onnx_path:
        # helpful debug listing
        print("No .onnx found under", model_dir)
        for root, dirs, files in os.walk(model_dir):
            print("DIR:", root, "FILES:", files[:50])
        raise FileNotFoundError(f"No .onnx file found in {model_dir} after extraction")

    print("ONNX model located at:", onnx_path)

    # 3) create Predictor (try TRT EP then fallback to CUDA)
    try:
        predictor = Predictor(onnx_path)
    except Exception as e:
        print("Failed to create Predictor with TensorRT EP:", e)
        print("Attempting to load with CUDAExecutionProvider only (fallback)...")
        try:
            # fallback: create InferenceSession with CUDA only
            session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            # wrap session into our Predictor-like object
            class SimplePredictor:
                def __init__(self, session):
                    self.session = session
                def predict(self, input_data):
                    inputs = {k: np.array(v, dtype=np.int64) for k, v in input_data.items()}
                    outputs = self.session.run(None, inputs)
                    return outputs[0].tolist()
            predictor = SimplePredictor(session)
            print("Loaded model with CUDA execution provider (fallback).")
        except Exception as e2:
            print("Fallback load also failed:", e2)
            raise

    # 4) Warmup: run a tiny dummy inference to trigger engine build if needed
    try:
        dummy_inputs = {
            "input_ids": [[101, 2009, 2003, 1037, 3944, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
        }
        print("Running warmup inference (dummy inputs) to prime model...")
        _ = predictor.predict(dummy_inputs)
        print("Warmup done.")
    except Exception as e:
        print("Warmup failed (this may be okay). Error:", e)

    return predictor

def predict_fn(input_data, model):
    return model.predict(input_data)
# ------------------ end new model_fn ------------------
