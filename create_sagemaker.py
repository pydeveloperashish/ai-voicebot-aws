# create_sagemaker.py  — idempotent deploy script
import boto3
import botocore
import time
import sys

# --- Config (fill these) ---
region = "ap-south-1"
account_id = "170722810688"
ecr_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/phi2-trt:latest"
s3_model_path = "s3://voiceai-s3-bucket-03/phi2-onnx-int8-model/model.tar.gz"

model_name = "phi2-trt-model"
endpoint_config_name = "phi2-trt-config"
endpoint_name = "phi2-trt-endpoint"
role_arn = "arn:aws:iam::170722810688:role/SageMakerExecutionRole"

sm = boto3.client("sagemaker", region_name=region)

def exists_model(name):
    try:
        sm.describe_model(ModelName=name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise

def exists_endpoint_config(name):
    try:
        sm.describe_endpoint_config(EndpointConfigName=name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise

def exists_endpoint(name):
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise

def wait_for_endpoint_status(name, target_statuses=("InService",), fail_statuses=("Failed",), timeout_minutes=30):
    deadline = time.time() + timeout_minutes*60
    while time.time() < deadline:
        resp = sm.describe_endpoint(EndpointName=name)
        status = resp["EndpointStatus"]
        print("  -> endpoint status:", status)
        if status in target_statuses:
            return status
        if status in fail_statuses:
            return status
        time.sleep(15)
    raise TimeoutError(f"Timeout waiting for endpoint {name} to reach {target_statuses}")

def delete_endpoint_safe(name):
    if not exists_endpoint(name):
        print("No endpoint to delete.")
        return
    print("Deleting existing endpoint:", name)
    sm.delete_endpoint(EndpointName=name)
    # also delete endpoint config (optional) - we'll keep but you can delete if you want
    # wait until endpoint is deleted
    deadline = time.time() + 15*60
    while time.time() < deadline:
        if not exists_endpoint(name):
            print("Endpoint deleted.")
            return
        print("  -> waiting for endpoint deletion...")
        time.sleep(10)
    raise TimeoutError("Timed out waiting for endpoint deletion")

# 1) Model create (idempotent)
print("📦 Ensuring model exists...")
if exists_model(model_name):
    print("Model already exists — skipping create_model")
else:
    print("Creating model:", model_name)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": ecr_image, "ModelDataUrl": s3_model_path},
        ExecutionRoleArn=role_arn,
    )
    print("Model created.")

# 2) Endpoint config create (idempotent)
print("⚙️ Ensuring endpoint config exists...")
if exists_endpoint_config(endpoint_config_name):
    print("Endpoint config already exists — skipping create_endpoint_config")
else:
    print("Creating endpoint config:", endpoint_config_name)
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.g5.xlarge",
                "InitialInstanceCount": 1,
            }
        ],
    )
    print("Endpoint config created.")

# 3) Endpoint create / reconcile
print("🚀 Reconciling endpoint state for:", endpoint_name)
if exists_endpoint(endpoint_name):
    desc = sm.describe_endpoint(EndpointName=endpoint_name)
    status = desc["EndpointStatus"]
    print("Existing endpoint status:", status)

    if status == "InService":
        print("✅ Endpoint already InService. Nothing to do.")
        sys.exit(0)

    if status in ("Creating", "Updating"):
        print("Endpoint is currently", status, " — waiting until complete or fail.")
        final = wait_for_endpoint_status(endpoint_name, target_statuses=("InService",), fail_statuses=("Failed",))
        if final == "InService":
            print("✅ Endpoint became InService.")
            sys.exit(0)
        else:
            print("Endpoint ended in status:", final, " — will delete and recreate.")
            delete_endpoint_safe(endpoint_name)

    if status == "Failed":
        print("Endpoint in Failed state — deleting and recreating.")
        delete_endpoint_safe(endpoint_name)

    # if we reach here, endpoint either deleted or not present; fall through to create

print("Creating endpoint:", endpoint_name)
sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)

print("⏳ Waiting for endpoint to be InService...")
final_status = wait_for_endpoint_status(endpoint_name, target_statuses=("InService",), fail_statuses=("Failed",), timeout_minutes=40)
if final_status == "InService":
    print("✅ Endpoint is live:", endpoint_name)
else:
    # fetch failure reason for diagnostics
    desc = sm.describe_endpoint(EndpointName=endpoint_name)
    raise RuntimeError("❌ Endpoint deployment failed: " + str(desc.get("FailureReason", desc)))
