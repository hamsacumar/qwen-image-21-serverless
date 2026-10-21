
import base64
import binascii
import io
import json
import os
import random
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

import boto3
import requests
import runpod
from PIL import Image


# ============================================================
# Configuration
# ============================================================

COMFY_URL = os.getenv(
    "COMFY_URL", "http://127.0.0.1:8188"
).rstrip("/")

WORKFLOW_PATH = os.getenv(
    "WORKFLOW_PATH", "/app/workflow_api.json"
)

EDIT_WORKFLOW_PATH = os.getenv(
    "EDIT_WORKFLOW_PATH", "/app/workflow_edit_api.json"
)

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# Temporary URL lifetime, in seconds.
S3_URL_EXPIRES_IN = int(
    os.getenv("S3_URL_EXPIRES_IN", "3600")
)

MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
MAX_INPUT_IMAGES = 2
MAX_OUTPUT_IMAGES = 4
COMFY_TIMEOUT_SECONDS = 900

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is not set")

if not 1 <= S3_URL_EXPIRES_IN <= 604800:
    raise RuntimeError(
        "S3_URL_EXPIRES_IN must be between 1 and 604800"
    )

s3 = boto3.client("s3", region_name=AWS_REGION)

session = requests.Session()


# ============================================================
# Workflow node IDs - preserve your supplied workflow IDs
# ============================================================

PROMPT_NODE = "459:471"
EDIT_PROMPT_NODE = "459:474"
SAMPLER_NODE = "459:458"
RESOLUTION_NODE = "13"

IMAGE_1_NODE = "470"
IMAGE_2_NODE = "475"

LATENT_SWITCH_NODE = "459:468"
EMPTY_LATENT_NODE = "459:456"

SAVE_IMAGE_NODE = "461"


# ============================================================
# Image size mappings
# ============================================================


IMAGE_SIZE_MAP = {
    # Square
    "square": "1:1 (Square)",
    "square_hd": "1:1 (Square)",

    # Portrait
    "portrait_2_3": "2:3 (Portrait Photo)",
    "portrait_3_4": "3:4 (Portrait Standard)",
    "portrait_4_3": "3:4 (Portrait Standard)",
    "portrait_16_9": "9:16 (Portrait Widescreen)",

    # Landscape
    "landscape_4_3": "4:3 (Standard)",
    "landscape_3_2": "3:2 (Photo)",
    "landscape_16_9": "16:9 (Widescreen)",
    "landscape_21_9": "21:9 (Ultrawide)",
}


ALLOWED_OUTPUT_FORMATS = {"png"}


# ============================================================
# General helpers
# ============================================================

def error(message, details=None):
    result = {"error": message}

    if details is not None:
        result["details"] = str(details)[:4000]

    return result


def load_workflow(path):
    with open(path, "r", encoding="utf-8") as file:
        workflow = json.load(file)

    if not isinstance(workflow, dict):
        raise ValueError(
            f"Workflow must be a JSON object: {path}"
        )

    return workflow


def get_image_size(data):
    image_size = data.get("image_size", "square")

    if not isinstance(image_size, str):
        raise ValueError("image_size must be a string")

    if image_size in IMAGE_SIZE_MAP:
        aspect_ratio = IMAGE_SIZE_MAP[image_size]
    elif image_size in IMAGE_SIZE_MAP.values():
        aspect_ratio = image_size
    else:
        allowed = ", ".join(IMAGE_SIZE_MAP.keys())
        raise ValueError(
            f"Unsupported image_size '{image_size}'. "
            f"Supported values: {allowed}, or a direct "
            "ResolutionSelector label."
        )

    megapixels = float(data.get("megapixels", 1))

    if not 0.25 <= megapixels <= 2:
        raise ValueError(
            "megapixels must be between 0.25 and 2"
        )

    return aspect_ratio, megapixels


def validate_image_bytes(image_bytes):
    if not image_bytes:
        raise ValueError("Input image is empty")

    if len(image_bytes) > MAX_INPUT_IMAGE_BYTES:
        raise ValueError(
            "Each input image must be 25 MB or smaller"
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(
            "Input does not contain a valid image"
        ) from exc

    return image_bytes


# ============================================================
# Input images: URL, data URI, or Base64
# ============================================================

def upload_image_bytes_to_comfy(image_bytes, filename):
    safe_filename = os.path.basename(filename)

    response = session.post(
        f"{COMFY_URL}/upload/image",
        files={
            "image": (
                safe_filename,
                image_bytes,
                "application/octet-stream",
            )
        },
        data={
            "type": "input",
            "overwrite": "true",
        },
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    uploaded_name = result.get("name")

    if not uploaded_name:
        raise RuntimeError(
            f"ComfyUI returned no uploaded image name: {result}"
        )

    return uploaded_name


def decode_base64_image(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Base64 image input must be a non-empty string"
        )

    value = value.strip()

    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(
            value, validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "Image input is not valid Base64 data"
        ) from exc

    return validate_image_bytes(image_bytes)


def download_image_url(url, index):
    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            f"image_urls[{index}] must be a non-empty URL"
        )

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"image_urls[{index}] must be a valid HTTP/HTTPS URL"
        )

    # Stream the download and stop if the size limit is exceeded.
    chunks = []
    total_bytes = 0

    with session.get(
        url.strip(),
        timeout=(15, 120),
        stream=True,
    ) as response:
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_INPUT_IMAGE_BYTES:
                    raise ValueError(
                        "Each input image must be 25 MB or smaller"
                    )
            except ValueError as exc:
                if "25 MB" in str(exc):
                    raise

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue

            total_bytes += len(chunk)

            if total_bytes > MAX_INPUT_IMAGE_BYTES:
                raise ValueError(
                    "Each input image must be 25 MB or smaller"
                )

            chunks.append(chunk)

    image_bytes = validate_image_bytes(b"".join(chunks))

    extension = os.path.splitext(parsed.path)[1].lower()
    if extension not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        extension = ".png"

    filename = f"runpod_input_{uuid.uuid4().hex}{extension}"

    return upload_image_bytes_to_comfy(
        image_bytes, filename
    )


def upload_base64_image_to_comfy(value, index):
    image_bytes = decode_base64_image(value)
    filename = f"runpod_input_{index}_{uuid.uuid4().hex}.png"

    return upload_image_bytes_to_comfy(
        image_bytes, filename
    )


def upload_input_image(value, index):
    """
    Accept an HTTP/HTTPS URL, data:image/...;base64,...,
    or raw Base64 image string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Input image {index} must be a non-empty string"
        )

    value = value.strip()

    if value.startswith(("http://", "https://")):
        return download_image_url(value, index)

    return upload_base64_image_to_comfy(value, index)


# ============================================================
# Wait for ComfyUI outputs
# ============================================================

def wait_for_images(prompt_id, timeout_seconds=COMFY_TIMEOUT_SECONDS):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        response = session.get(
            f"{COMFY_URL}/history/{prompt_id}",
            timeout=30,
        )
        response.raise_for_status()

        history = response.json()
        prompt_history = history.get(prompt_id)

        if prompt_history:
            status = prompt_history.get("status", {})
            status_string = status.get("status_str", "")

            if status_string == "error":
                messages = status.get("messages", [])
                return error(
                    "ComfyUI workflow failed",
                    messages or status,
                )

            outputs = prompt_history.get("outputs", {})

            # Prefer the configured Save Image node to avoid
            # collecting previews or unrelated intermediate images.
            if SAVE_IMAGE_NODE in outputs:
                selected_outputs = {
                    SAVE_IMAGE_NODE: outputs[SAVE_IMAGE_NODE]
                }
            else:
                selected_outputs = outputs

            images = []

            for node_output in selected_outputs.values():
                for image_info in node_output.get("images", []):
                    filename = image_info.get("filename")
                    if not filename:
                        continue

                    image_response = session.get(
                        f"{COMFY_URL}/view",
                        params={
                            "filename": filename,
                            "subfolder": image_info.get(
                                "subfolder", ""
                            ),
                            "type": image_info.get(
                                "type", "output"
                            ),
                        },
                        timeout=120,
                    )
                    image_response.raise_for_status()

                    images.append({
                        "filename": filename,
                        "mime_type": image_response.headers.get(
                            "Content-Type", "image/png"
                        ),
                        "image_bytes": image_response.content,
                    })

            if images:
                return {"images": images}

            if status_string == "success":
                return error(
                    "ComfyUI completed without producing an image",
                    {
                        "prompt_id": prompt_id,
                        "status": status,
                        "output_nodes": list(outputs.keys()),
                    },
                )

            time.sleep(2)

            return error(
                "Timed out waiting for ComfyUI",
                f"prompt_id={prompt_id}, timeout={timeout_seconds}s",
            )


# ============================================================
# S3 uploads
# ============================================================

def upload_result_to_s3(
    image_bytes, job_id, filename, mime_type
):
    extension = os.path.splitext(filename)[1] or ".png"
    s3_key = (
        f"images/{job_id}/"
        f"{uuid.uuid4().hex}{extension}"
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType=mime_type,
    )

    # Temporary signed URL; it does not require a public bucket.
    signed_url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": s3_key,
        },
        ExpiresIn=S3_URL_EXPIRES_IN,
    )

    return {
        "filename": filename,
        "mime_type": mime_type,
        "s3_bucket": S3_BUCKET,
        "s3_key": s3_key,
        "s3_uri": f"s3://{S3_BUCKET}/{s3_key}",
        "url": signed_url,
        "url_expires_in": S3_URL_EXPIRES_IN,
    }


# ============================================================
# RunPod handler
# ============================================================

def handler(job):
    data = job.get("input") or {}

    if not isinstance(data, dict):
        return error("input must be a JSON object")

    # Explicit mode takes priority. Otherwise infer it from
    # image_urls or the legacy image field.
    requested_mode = data.get("mode")

    if requested_mode is None:
        mode = (
            "edit"
            if data.get("image_urls") is not None
            or data.get("image") is not None
            else "generate"
        )
    else:
        mode = requested_mode

    if mode not in ("generate", "edit"):
        return error("mode must be 'generate' or 'edit'")

    prompt = data.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return error("Please provide a non-empty prompt")

    prompt = prompt.strip()

    try:
        steps = int(
            data.get(
                "num_inference_steps",
                data.get("steps", 28),
            )
        )
        guidance_scale = float(
            data.get("guidance_scale", 4.0)
        )
        num_images = int(data.get("num_images", 1))
        output_format = str(
            data.get("output_format", "png")
        ).lower()

        seed_value = data.get("seed")
        seed = (
            random.randint(0, 2**64 - 1)
            if seed_value is None
            else int(seed_value)
        )

        aspect_ratio, megapixels = get_image_size(data)

    except (TypeError, ValueError, OverflowError) as exc:
        return error("Invalid input parameter", exc)

    if not 1 <= steps <= 60:
        return error(
            "num_inference_steps must be between 1 and 60"
        )

    if not 0 <= seed <= 2**64 - 1:
        return error("seed must be between 0 and 2^64-1")

    if not 0 <= guidance_scale <= 30:
        return error("guidance_scale must be between 0 and 30")

    if not 1 <= num_images <= MAX_OUTPUT_IMAGES:
        return error(
            f"num_images must be between 1 and {MAX_OUTPUT_IMAGES}"
        )

    if output_format not in ALLOWED_OUTPUT_FORMATS:
        return error(
            "Only output_format='png' is supported"
        )

    # Accepted for API compatibility. These are not implemented
    # unless the workflow/container has corresponding support.
    enable_safety_checker = data.get(
        "enable_safety_checker", True
    )
    acceleration = data.get("acceleration", "regular")

    if not isinstance(enable_safety_checker, bool):
        return error("enable_safety_checker must be a boolean")

    if not isinstance(acceleration, str) or not acceleration.strip():
        return error("acceleration must be a non-empty string")

    try:
        # ----------------------------------------------------
        # Load and configure the selected workflow
        # ----------------------------------------------------

        if mode == "generate":
            workflow = load_workflow(WORKFLOW_PATH)

            if PROMPT_NODE not in workflow:
                return error(
                    f"Generate prompt node '{PROMPT_NODE}' "
                    "was not found in workflow_api.json"
                )

            workflow[PROMPT_NODE]["inputs"]["prompt"] = prompt

        else:
            workflow = load_workflow(EDIT_WORKFLOW_PATH)

            if EDIT_PROMPT_NODE not in workflow:
                return error(
                    f"Edit prompt node '{EDIT_PROMPT_NODE}' "
                    "was not found in workflow_edit_api.json"
                )

            if IMAGE_1_NODE not in workflow:
                return error(
                    f"Image node '{IMAGE_1_NODE}' "
                    "was not found in the edit workflow"
                )

            image_urls = data.get("image_urls")

            # Legacy Base64 API: {"mode":"edit","image":"..."}
            if image_urls is None:
                legacy_image = data.get("image")

                if not legacy_image:
                    return error(
                        "Edit mode requires image_urls or image"
                    )

                image_urls = [legacy_image]

            if (
                not isinstance(image_urls, list)
                or not 1 <= len(image_urls) <= MAX_INPUT_IMAGES
            ):
                return error(
                    "image_urls must be a list containing "
                    "one or two images"
                )

            # Upload each input image to ComfyUI.
            image_1_name = upload_input_image(
                image_urls[0], 1
            )
            workflow[IMAGE_1_NODE]["inputs"]["image"] = image_1_name

            if len(image_urls) == 2:
                if IMAGE_2_NODE not in workflow:
                    return error(
                        f"Second image node '{IMAGE_2_NODE}' "
                        "was not found in the edit workflow"
                    )

                image_2_name = upload_input_image(
                    image_urls[1], 2
                )
                workflow[IMAGE_2_NODE]["inputs"]["image"] = (
                    image_2_name
                )

            workflow[EDIT_PROMPT_NODE]["inputs"]["prompt"] = prompt

            if LATENT_SWITCH_NODE in workflow:
                workflow[LATENT_SWITCH_NODE]["inputs"]["switch"] = (
                    False
                )

        # ----------------------------------------------------
        # Shared sampler and resolution settings
        # ----------------------------------------------------

        if SAMPLER_NODE not in workflow:
            return error(
                f"Sampler node '{SAMPLER_NODE}' was not found"
            )

        workflow[SAMPLER_NODE]["inputs"]["seed"] = seed
        workflow[SAMPLER_NODE]["inputs"]["steps"] = steps
        workflow[SAMPLER_NODE]["inputs"]["cfg"] = guidance_scale

        if RESOLUTION_NODE in workflow:
            workflow[RESOLUTION_NODE]["inputs"]["aspect_ratio"] = (
                aspect_ratio
            )
            workflow[RESOLUTION_NODE]["inputs"]["megapixels"] = (
                megapixels
            )

        # This changes the batch only when the workflow contains
        # this exact EmptyLatentImage node.
        if EMPTY_LATENT_NODE in workflow:
            workflow[EMPTY_LATENT_NODE]["inputs"]["batch_size"] = (
                num_images
            )
        elif num_images > 1:
            return error(
                "This workflow has no configured empty latent "
                "batch node. Multiple-image generation requires "
                "a workflow that supports batch_size."
            )

        if SAVE_IMAGE_NODE in workflow:
            workflow[SAVE_IMAGE_NODE]["inputs"]["format"] = (
                output_format
            )

        # ----------------------------------------------------
        # Submit the workflow to ComfyUI
        # ----------------------------------------------------

        response = session.post(
            f"{COMFY_URL}/prompt",
            json={
                "prompt": workflow,
                "client_id": str(uuid.uuid4()),
            },
            timeout=60,
        )
        response.raise_for_status()

        prompt_id = response.json().get("prompt_id")

        if not prompt_id:
            return error(
                "ComfyUI did not return a prompt_id",
                response.text[:2000],
            )

        # ----------------------------------------------------
        # Collect all generated images
        # ----------------------------------------------------

        result = wait_for_images(prompt_id)

        if "error" in result:
            return result

        generated_images = result["images"]

        if not generated_images:
            return error(
                "ComfyUI finished without returning any images"
            )

        # ----------------------------------------------------
        # Upload every result to S3
        # ----------------------------------------------------

        job_id = job.get("id") or str(uuid.uuid4())
        image_results = []

        for generated in generated_images:
            image_results.append(
                upload_result_to_s3(
                    image_bytes=generated["image_bytes"],
                    job_id=job_id,
                    filename=generated["filename"],
                    mime_type=generated["mime_type"],
                )
            )

        return {
            "status": "COMPLETED",
            "mode": mode,
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "num_images_requested": num_images,
            "num_images_generated": len(image_results),
            "image_size": data.get("image_size", "square"),
            "megapixels": megapixels,
            "output_format": output_format,
            "enable_safety_checker_requested": enable_safety_checker,
            "acceleration_requested": acceleration,
            "s3_bucket": S3_BUCKET,
            "images": image_results,
            "prompt_id": prompt_id,
        }

    except requests.RequestException as exc:
        print(
            f"HTTP/request error: {exc}",
            flush=True,
        )
        return error(
            "Could not communicate with ComfyUI or download an image",
            exc,
        )

    except Exception as exc:
        print(
            f"Unexpected handler error: {exc}",
            flush=True,
        )
        return error("Unexpected handler error", exc)


# ============================================================
# Configure persistent Qwen model paths
# ============================================================

def configure_model_paths():
    candidates = [
        "/runpod-volume/qwen-models",
        "/workspace/qwen-models",
    ]

    model_root = next(
        (
            path
            for path in candidates
            if os.path.isdir(path)
        ),
        None,
    )

    if model_root is None:
        raise RuntimeError(
            "Qwen model directory not found. Checked: "
            + ", ".join(candidates)
        )

    required_files = {
        "diffusion_models": "qwen_image_2.1_int8_convrot.safetensors",
        "text_encoders": "qwen3vl_8b_int8_convrot.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
    }

    for folder, filename in required_files.items():
        path = os.path.join(model_root, folder, filename)

        if not os.path.isfile(path):
            raise RuntimeError(
                f"Missing model file: {path}. "
                "Check your persistent volume folder structure."
            )

    config = f"""\
qwen_persistent:
  base_path: "{model_root}/"
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  vae: vae/
"""

    config_path = "/opt/ComfyUI/extra_model_paths.yaml"

    with open(config_path, "w", encoding="utf-8") as file:
        file.write(config)

    print(
        f"Configured Qwen model paths: {model_root}",
        flush=True,
    )


# ============================================================
# Start ComfyUI before accepting RunPod jobs
# ============================================================

def start_comfyui():
    configure_model_paths()

    print("Starting ComfyUI...", flush=True)

    comfy_process = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
        ],
        cwd="/opt/ComfyUI",
    )

    deadline = time.time() + 600

    while time.time() < deadline:
        if comfy_process.poll() is not None:
            raise RuntimeError(
                "ComfyUI exited during startup with code "
                f"{comfy_process.returncode}. Check worker logs."
            )

        try:
            response = session.get(
                f"{COMFY_URL}/system_stats",
                timeout=5,
            )

            if response.status_code == 200:
                print("ComfyUI is ready!", flush=True)
                return comfy_process

        except requests.RequestException:
            pass

        time.sleep(2)

    comfy_process.terminate()

    raise RuntimeError(
        "ComfyUI did not become ready within 600 seconds. "
        "Check worker logs for startup errors."
    )


# Start ComfyUI first, then accept RunPod Serverless jobs.
comfy_process = start_comfyui()

runpod.serverless.start({"handler": handler})
