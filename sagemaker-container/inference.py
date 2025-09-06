import os
import onnxruntime as ort
import numpy as np

class Predictor:
    def __init__(self, model_path):
        providers = [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CUDAExecutionProvider",
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)

    def predict(self, input_data):
        # Example: input_data = {"input_ids": [[101, 2009, 2003, 1037, 3944, 102]], "attention_mask": [[1,1,1,1,1,1]]}
        inputs = {k: np.array(v, dtype=np.int64) for k, v in input_data.items()}
        outputs = self.session.run(None, inputs)
        return outputs[0].tolist()

model = None

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "content/phi2-onnx/model-int8.onnx")
    predictor = Predictor(model_path)

    # 🔥 Warmup: run a fake inference to trigger TensorRT engine build
    dummy_inputs = {
        "input_ids": [[101, 2009, 2003, 1037, 3944, 102]],  # [CLS] it is a test [SEP]
        "attention_mask": [[1, 1, 1, 1, 1, 1]],
    }
    _ = predictor.predict(dummy_inputs)

    return predictor

def predict_fn(input_data, model):
    return model.predict(input_data)
