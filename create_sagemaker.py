# create_sagemaker.py  — idempotent deploy script (enhanced)
import boto3
import botocore
import time
import sys
import json

# --- Config (fill these) ---
region = "ap-south-1"
account_id = "170722810688"
ecr_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/phi2-trt:latest"
s3_model_path = "s3://voiceai-s3-bucket-03/phi2-onnx-int8-model/model.tar.gz"

model_name = "phi2-trt-model"
endpoint_config_name = "phi2-trt-config"
endpoint_name = "phi2-trt-endpoint"
role_arn = "arn:aws:iam::170722810688:role/SageMakerExecutionRole"

# Clients
sm = boto3.client("sagemaker", region_name=region)
cw = boto3.client("logs", region_name=region)

# Tunables
POLL_INTERVAL = 15  # seconds between status checks
DELETE_WAIT_SECONDS = 15 * 60
CREATE_TIMEOUT_MINUTES = 40


def exists_model(name):
    try:
        sm.describe_model(ModelName=name)
        return True
    except botocore.exceptions.ClientError as e:
        # SageMaker raises ValidationException when resource not found for describe_model
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def exists_endpoint_config(name):
    try:
        sm.describe_endpoint_config(EndpointConfigName=name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def exists_endpoint(name):
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def wait_for_endpoint_status(name, target_statuses=("InService",), fail_statuses=("Failed",), timeout_minutes=30):
    deadline = time.time() + timeout_minutes * 60
    last_status = None
    while time.time() < deadline:
        try:
            resp = sm.describe_endpoint(EndpointName=name)
        except botocore.exceptions.ClientError as e:
            # If endpoint does not exist yet, sleep and retry
            if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
                print("  -> endpoint not found yet; waiting...")
                time.sleep(POLL_INTERVAL)
                continue
            raise
        status = resp.get("EndpointStatus")
        if status != last_status:
            print(f"  -> endpoint status: {status}")
            last_status = status
        if status in target_statuses:
            return status
        if status in fail_statuses:
            return status
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Timeout waiting for endpoint {name} to reach {target_statuses}")


def delete_endpoint_safe(name, wait_seconds=DELETE_WAIT_SECONDS):
    if not exists_endpoint(name):
        print("No endpoint to delete.")
        return
    print("Deleting existing endpoint:", name)
    sm.delete_endpoint(EndpointName=name)
    # wait until endpoint is deleted
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not exists_endpoint(name):
            print("Endpoint deleted.")
            return
        print("  -> waiting for endpoint deletion...")
        time.sleep(10)
    raise TimeoutError("Timed out waiting for endpoint deletion")


def describe_failure_reason(name):
    try:
        desc = sm.describe_endpoint(EndpointName=name)
        return desc.get("FailureReason", json.dumps(desc, default=str))
    except botocore.exceptions.ClientError as e:
        return f"Could not describe endpoint: {e}"


def tail_cloudwatch_logs_for_endpoint(name, limit_streams=2, lines=200):
    log_group = f"/aws/sagemaker/Endpoints/{name}"
    try:
        streams_resp = cw.describe_log_streams(
            logGroupName=log_group, orderBy="LastEventTime", descending=True, limit=limit_streams
        )
        streams = streams_resp.get("logStreams", [])
        if not streams:
            print("No CloudWatch log streams found for", log_group)
            return
        for s in streams:
            stream_name = s["logStreamName"]
            print("\n--- CloudWatch log stream:", stream_name, "---\n")
            events = cw.get_log_events(logGroupName=log_group, logStreamName=stream_name, limit=lines, startFromHead=False)
            for e in events.get("events", []):
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e["timestamp"] / 1000.0))
                # print raw message (many messages already contain newline)
                print(f"[{ts}] {e['message']}", end="")
    except cw.exceptions.ResourceNotFoundException:
        print("CloudWatch log group not found:", log_group)
    except Exception as e:
        print("Error fetching CloudWatch logs:", e)


def ensure_model_and_config():
    # create model if missing
    if exists_model(model_name):
        print("Model already exists — skipping create_model")
    else:
        print("Creating model:", model_name)
        try:
            sm.create_model(
                ModelName=model_name,
                PrimaryContainer={"Image": ecr_image, "ModelDataUrl": s3_model_path},
                ExecutionRoleArn=role_arn,
            )
            print("Model created:", model_name)
        except botocore.exceptions.ClientError as e:
            print("Error creating model:", e)
            raise

    # create endpoint config if missing
    if exists_endpoint_config(endpoint_config_name):
        print("Endpoint config already exists — skipping create_endpoint_config")
    else:
        print("Creating endpoint config:", endpoint_config_name)
        try:
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
            print("Endpoint config created:", endpoint_config_name)
        except botocore.exceptions.ClientError as e:
            print("Error creating endpoint config:", e)
            raise


def reconcile_endpoint():
    print("🚀 Reconciling endpoint state for:", endpoint_name)
    # If exists, check status & react accordingly
    if exists_endpoint(endpoint_name):
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc.get("EndpointStatus")
        print("Existing endpoint status:", status)

        if status == "InService":
            print("✅ Endpoint already InService. Nothing to do.")
            return "InService"

        if status in ("Creating", "Updating"):
            print("Endpoint is currently", status, " — waiting until complete or fail.")
            final = wait_for_endpoint_status(endpoint_name, target_statuses=("InService",), fail_statuses=("Failed",), timeout_minutes=CREATE_TIMEOUT_MINUTES)
            if final == "InService":
                print("✅ Endpoint became InService.")
                return "InService"
            else:
                print("Endpoint ended in status:", final, " — will delete and recreate.")
                delete_endpoint_safe(endpoint_name)

        if status == "Failed":
            print("Endpoint in Failed state — deleting and recreating.")
            print("FailureReason:", desc.get("FailureReason"))
            delete_endpoint_safe(endpoint_name)

        # other statuses fall through to create

    # Create endpoint
    print("Creating endpoint:", endpoint_name)
    try:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    except botocore.exceptions.ClientError as e:
        print("Error creating endpoint:", e)
        raise

    print("⏳ Waiting for endpoint to be InService...")
    final_status = wait_for_endpoint_status(endpoint_name, target_statuses=("InService",), fail_statuses=("Failed",), timeout_minutes=CREATE_TIMEOUT_MINUTES)
    if final_status == "InService":
        print("✅ Endpoint is live:", endpoint_name)
        return "InService"
    else:
        reason = describe_failure_reason(endpoint_name)
        print("❌ Endpoint deployment failed. FailureReason:")
        print(reason)
        print("\n--- Tail CloudWatch logs for quick diagnostics ---")
        tail_cloudwatch_logs_for_endpoint(endpoint_name, limit_streams=3, lines=500)
        raise RuntimeError("❌ Endpoint deployment failed: " + str(reason))


if __name__ == "__main__":
    try:
        print("📦 Ensuring model and endpoint config exist...")
        ensure_model_and_config()

        result = reconcile_endpoint()
        print("Done. Result:", result)
    except Exception as e:
        print("Deployment failed with exception:", e)
        # show CloudWatch logs as last resort if endpoint exists but failed
        if exists_endpoint(endpoint_name):
            print("\n--- Additional CloudWatch logs (post-failure) ---")
            tail_cloudwatch_logs_for_endpoint(endpoint_name, limit_streams=3, lines=500)
        sys.exit(1)
