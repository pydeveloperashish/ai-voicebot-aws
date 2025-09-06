import json
import os
from flask import Flask, request, Response, jsonify
from inference import model_fn, predict_fn

app = Flask(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")
print("Loading model from:", MODEL_DIR)
try:
    model = model_fn(MODEL_DIR)
    print("Model loaded successfully.")
except Exception as e:
    # If model load fails, we still want the container to start so that logs show tracebacks.
    # SageMaker health check may fail, but logs will include the exception details.
    print("Model loading failed at startup:", e)
    model = None

@app.route("/ping", methods=["GET"])
def ping():
    # Basic health check that returns 200 if model loaded, else 500
    try:
        if model is not None:
            return Response("pong", status=200)
        else:
            return Response("model_not_loaded", status=500)
    except Exception as e:
        return Response("error", status=500)

@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        if request.content_type and "application/json" in request.content_type:
            payload = request.get_json(force=True)
        else:
            # attempt to decode whatever body is present
            try:
                payload = json.loads(request.data.decode("utf-8") or "{}")
            except:
                payload = {}

        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object/dict as input"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        result = predict_fn(payload, model)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # For SageMaker this builtin server is acceptable; for production replace with gunicorn/uvicorn
    app.run(host="0.0.0.0", port=port)
