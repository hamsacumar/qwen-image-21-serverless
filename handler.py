import os
import json
import time
import uuid
import base64
import binascii
import subprocess
import mimetypes

import boto3
import requests
import runpod

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "/app/workflow_api.json"
EDIT_WORKFLOW_PATH = "/app/workflow_edit_api.json"

MODEL_PATH = (
    "/runpod-volume/qwen-models"
    if os.path.isdir("/runpod-volume/qwen-models")
    else "/workspace/qwen-models"
)
CONFIG_PATH = "/app/extra_model_paths.yaml"

PROMPT_NODE = "459:471"
EDIT_PROMPT_NODE = "459:474"
SAMPLER_NODE = "459:458"
RESOLUTION_NODE = "13"
SAVE_NODE = "461"

session = requests.Session()

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is not set")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
)

def decode_image(value, field_name):
    """Decode a Base64 image or data URI into bytes."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Please provide {field_name} as a Base64 image")

    encoded = value.strip()

    if encoded.startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
        except ValueError:
            raise ValueError(f"Invalid data URI for {field_name}")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"{field_name} is not valid Base64")

    if not image_bytes:
        raise ValueError(f"{field_name} is empty")

    return image_bytes


def upload_image(image_value, field_name, filename):
    """Upload a Base64 image to ComfyUI and return its input filename."""
    image_bytes = decode_image(image_value, field_name)

    response = session.post(
        f"{COMFY_URL}/upload/image",
        data={"overwrite": "true"},
        files={
            "image": (
                filename,
                image_bytes,
                mimetypes.guess_type(filename)[0] or "image/png",
            )
        },
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    uploaded_name = result.get("name")

    if not uploaded_name:
        raise RuntimeError(
            f"ComfyUI did not return a filename for {field_name}"
        )

    subfolder = result.get("subfolder", "")
    return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name


def wait_for_image(prompt_id, timeout_seconds=900):
    """Wait for ComfyUI to finish and return the saved image bytes."""
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = session.get(
            f"{COMFY_URL}/history/{prompt_id}",
            timeout=30,
        )
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})

            if status.get("status_str") == "error":
                return {
                    "error": "ComfyUI generation failed",
                    "details": status.get("messages", []),
                }

            if status.get("completed"):
                images = item.get("outputs", {}).get(
                    SAVE_NODE, {}
                ).get("images", [])

                if not images:
                    return {"error": "No output image found"}

                info = images[0]
                image_response = session.get(
                    f"{COMFY_URL}/view",
                    params={
                        "filename": info["filename"],
                        "subfolder": info.get("subfolder", ""),
                        "type": info.get("type", "output"),
                    },
                    timeout=120,
                )
                image_response.raise_for_status()

                image_bytes = image_response.content

                return {
                    "image_bytes": image_bytes,
                    "mime_type": "image/png",
                    "filename": info["filename"],
                }

        time.sleep(2)

    return {"error": "Generation timed out"}


def upload_bytes_to_s3(image_bytes, s3_key, content_type="image/png"):
    """Upload generated image bytes directly to S3."""

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType=content_type,
    )

    return s3_key

def handler(job):
    print("========== HANDLER STARTED ==========", flush=True)

    data = job.get("input") or {}
    mode = data.get("mode", "generate")

    print(f"DEBUG job_id: {job.get('id')}", flush=True)
    print(f"DEBUG mode: {mode!r}", flush=True)
    print(f"DEBUG input keys: {list(data.keys())}", flush=True)

    image = data.get("image")

    print(f"DEBUG image type: {type(image).__name__}", flush=True)
    print(
        f"DEBUG image length: "
        f"{len(image) if isinstance(image, str) else 'N/A'}",
        flush=True,
    )
    print(f"DEBUG image present: {bool(image)}", flush=True)

    if mode not in ("generate", "edit"):
        return {"error": "mode must be 'generate' or 'edit'"}

    if mode == "edit" and not image:
        return {
            "error": "Edit mode requires a non-empty Base64 image",
            "received_input_keys": list(data.keys()),
        }

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Please provide a non-empty prompt"}

    try:
        steps = int(data.get("steps", 25))
        seed = int(data.get("seed", 42))
        megapixels = float(data.get("megapixels", 1))
    except (TypeError, ValueError):
        return {"error": "steps, seed and megapixels must be numeric"}

    aspect_ratio = data.get("aspect_ratio", "1:1 (Square)")

    if not 1 <= steps <= 60:
        return {"error": "steps must be between 1 and 60"}
    if not 0 <= seed <= 2**64 - 1:
        return {"error": "seed is outside the supported range"}
    if not 0.25 <= megapixels <= 2:
        return {"error": "megapixels must be between 0.25 and 2"}

    try:
        if mode == "generate":
            with open(WORKFLOW_PATH) as f:
                workflow = json.load(f)

            workflow[PROMPT_NODE]["inputs"]["prompt"] = prompt

        else:
            with open(EDIT_WORKFLOW_PATH) as f:
                workflow = json.load(f)

            workflow["470"]["inputs"]["image"] = upload_image(
                image,
                "image",
                "original.png",
            )

            workflow[EDIT_PROMPT_NODE]["inputs"]["prompt"] = prompt

        workflow[SAMPLER_NODE]["inputs"]["seed"] = seed
        workflow[SAMPLER_NODE]["inputs"]["steps"] = steps
        workflow[RESOLUTION_NODE]["inputs"]["aspect_ratio"] = aspect_ratio
        workflow[RESOLUTION_NODE]["inputs"]["megapixels"] = megapixels

        print(
            f"DEBUG submitting ComfyUI workflow: mode={mode}",
            flush=True,
        )

        response = session.post(
            f"{COMFY_URL}/prompt",
            json={
                "prompt": workflow,
                "client_id": str(uuid.uuid4()),
            },
            timeout=60,
        )
        response.raise_for_status()

        prompt_id = response.json()["prompt_id"]
        print(f"DEBUG ComfyUI prompt_id: {prompt_id}", flush=True)

        result = wait_for_image(prompt_id)

        if "error" in result:
            print(f"DEBUG generation error: {result}", flush=True)
            return result

        # Get generated image bytes
        image_bytes = result.pop("image_bytes")

        # Use the RunPod job ID to create a unique S3 path
        job_id = job.get("id", str(uuid.uuid4()))

        s3_key = f"images/{job_id}/{result['filename']}"

        print(
            f"DEBUG uploading generated image to S3: "
            f"s3://{S3_BUCKET}/{s3_key}",
            flush=True,
        )

        upload_bytes_to_s3(
            image_bytes,
            s3_key,
            result.get("mime_type", "image/png"),
        )

        result.update({
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "s3_bucket": S3_BUCKET,
            "s3_key": s3_key,
        })

        print("DEBUG S3 upload completed successfully", flush=True)
        print("DEBUG generation completed successfully", flush=True)

        return result

    except ValueError as exc:
        print(f"DEBUG validation error: {exc}", flush=True)
        return {"error": str(exc)}

    except requests.RequestException as exc:
        print(f"DEBUG request error: {exc}", flush=True)
        return {
            "error": "Could not communicate with ComfyUI",
            "details": str(exc),
        }

    except Exception as exc:
        print(f"DEBUG unexpected error: {exc}", flush=True)
        return {
            "error": "Unexpected handler error",
            "details": str(exc),
        }


print("=== STARTUP DIAGNOSTICS ===", flush=True)
print(
    f"/runpod-volume exists: {os.path.isdir('/runpod-volume')}",
    flush=True,
)
print(
    f"/runpod-volume/qwen-models exists: "
    f"{os.path.isdir('/runpod-volume/qwen-models')}",
    flush=True,
)
print(
    f"/workspace/qwen-models exists: "
    f"{os.path.isdir('/workspace/qwen-models')}",
    flush=True,
)
print(f"Selected MODEL_PATH: {MODEL_PATH}", flush=True)

if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(
        f"Models volume not found: {MODEL_PATH}"
    )

with open(CONFIG_PATH, "w") as f:
    f.write(
        "qwen_persistent:\n"
        f"  base_path: {MODEL_PATH}/\n"
        "  diffusion_models: diffusion_models/\n"
        "  text_encoders: text_encoders/\n"
        "  vae: vae/\n"
    )


print("=== STARTING COMFYUI ===", flush=True)

comfy_process = subprocess.Popen(
    [
        "python", "-u", "/opt/ComfyUI/main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--extra-model-paths-config", CONFIG_PATH,
    ],
    cwd="/opt/ComfyUI",
    stdout=None,
    stderr=None,
)


for attempt in range(180):
    exit_code = comfy_process.poll()

    if exit_code is not None:
        raise RuntimeError(
            f"ComfyUI exited during startup "
            f"with exit code {exit_code}"
        )

    try:
        response = session.get(
            f"{COMFY_URL}/system_stats",
            timeout=3,
        )
        if response.ok:
            print("=== COMFYUI IS READY ===", flush=True)
            break
    except requests.RequestException:
        pass

    time.sleep(2)
else:
    comfy_process.terminate()
    raise RuntimeError("ComfyUI startup timed out")

runpod.serverless.start({"handler": handler})
