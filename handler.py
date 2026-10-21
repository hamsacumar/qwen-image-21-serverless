import base64
import io
import json
import os
import random
import time
import uuid
from urllib.parse import urlparse
import subprocess
import sys
import boto3
import requests
import runpod
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

WORKFLOW_PATH = os.getenv(
    "WORKFLOW_PATH",
    "/app/workflow_api.json",
)

EDIT_WORKFLOW_PATH = os.getenv(
    "EDIT_WORKFLOW_PATH",
    "/app/workflow_edit_api.json",
)

COMFY_INPUT_DIR = os.getenv(
    "COMFY_INPUT_DIR",
    "/workspace/ComfyUI/input",
)

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is not set")

s3 = boto3.client("s3", region_name=AWS_REGION)

session = requests.Session()


# --------------------------------------------------
# Workflow node IDs
# These IDs are from your supplied workflow.
# --------------------------------------------------

PROMPT_NODE = "459:471"       # Generate workflow prompt node
EDIT_PROMPT_NODE = "459:474"  # Edit workflow prompt node
SAMPLER_NODE = "459:458"
RESOLUTION_NODE = "13"

# Image input nodes in your supplied edit workflow
IMAGE_1_NODE = "470"
IMAGE_2_NODE = "475"

# Empty-latent switch in the supplied workflow
LATENT_SWITCH_NODE = "459:468"

# Image batch size in the supplied workflow
EMPTY_LATENT_NODE = "459:456"

SAVE_IMAGE_NODE = "461"


# Common API image-size names mapped to likely selector labels.
# Confirm these labels against the ResolutionSelector dropdown
# in your installed ComfyUI workflow.
IMAGE_SIZE_MAP = {
    "square": "1:1 (Square)",
    "square_hd": "1:1 (Square)",
    "portrait_4_3": "3:4 (Portrait)",
    "portrait_3_4": "3:4 (Portrait)",
    "portrait_16_9": "9:16 (Portrait)",
    "landscape_4_3": "4:3 (Landscape)",
    "landscape_3_2": "3:2 (Landscape)",
    "landscape_16_9": "16:9 (Landscape)",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_workflow(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def upload_image_bytes_to_comfy(image_bytes, filename):
    """
    Upload image bytes to ComfyUI's input directory using
    ComfyUI's /upload/image endpoint.
    """
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
            f"ComfyUI did not return an uploaded image name: {result}"
        )

    return uploaded_name


def download_image_url(url, index):
    """
    Download an image URL and upload the image to ComfyUI.
    Only HTTP/HTTPS URLs are accepted.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Each image_urls item must be a non-empty URL")

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("image_urls must contain valid HTTP/HTTPS URLs")

    response = session.get(
        url,
        timeout=(15, 120),
        stream=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    image_bytes = response.content

    if not image_bytes:
        raise ValueError(f"Image URL {index} returned an empty file")

    if len(image_bytes) > 25 * 1024 * 1024:
        raise ValueError("Each input image must be 25 MB or smaller")

    # Verify that the response body is actually an image.
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(
            f"image_urls[{index}] did not return a valid image"
        ) from exc

    extension = os.path.splitext(parsed.path)[1].lower()
    if extension not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        extension = ".png"

    filename = f"runpod_input_{uuid.uuid4().hex}{extension}"

    return upload_image_bytes_to_comfy(image_bytes, filename)


def decode_base64_image(image_value):
    """
    Also supports the old API's Base64 image input.
    Accepts raw Base64 or a data:image/...;base64,... string.
    """
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("image must contain Base64 image data")

    value = image_value.strip()

    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("image is not valid Base64 data") from exc

    if not image_bytes:
        raise ValueError("Decoded image is empty")

    if len(image_bytes) > 25 * 1024 * 1024:
        raise ValueError("Input image must be 25 MB or smaller")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise ValueError("Base64 data does not contain a valid image") from exc

    return image_bytes


def upload_base64_image_to_comfy(image_value, index):
    image_bytes = decode_base64_image(image_value)
    filename = f"runpod_input_{uuid.uuid4().hex}.png"
    return upload_image_bytes_to_comfy(image_bytes, filename)


def wait_for_image(prompt_id, timeout_seconds=900):
    """
    Poll ComfyUI history until the workflow finishes.
    Return the first generated image.
    """
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
                return {
                    "error": "ComfyUI workflow failed",
                    "details": status,
                }

            outputs = prompt_history.get("outputs", {})

            for node_output in outputs.values():
                images = node_output.get("images", [])

                for image_info in images:
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    image_type = image_info.get("type", "output")

                    if not filename:
                        continue

                    image_response = session.get(
                        f"{COMFY_URL}/view",
                        params={
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": image_type,
                        },
                        timeout=120,
                    )
                    image_response.raise_for_status()

                    return {
                        "filename": filename,
                        "mime_type": image_response.headers.get(
                            "Content-Type", "image/png"
                        ),
                        "image_bytes": image_response.content,
                    }

        time.sleep(2)

    return {
        "error": "Timed out waiting for ComfyUI",
        "prompt_id": prompt_id,
    }


def upload_result_to_s3(image_bytes, job_id, filename, mime_type):
    extension = os.path.splitext(filename)[1] or ".png"
    s3_key = f"images/{job_id}/{uuid.uuid4().hex}{extension}"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType=mime_type,
    )

    return s3_key


def get_image_size(data):
    """
    Return the ResolutionSelector aspect ratio and megapixel value.

    image_size may be an API-style value such as 'landscape_4_3',
    or a direct ResolutionSelector label such as '1:1 (Square)'.
    """
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
            f"Supported API values: {allowed}"
        )

    megapixels = float(data.get("megapixels", 1))

    if not 0.25 <= megapixels <= 2:
        raise ValueError("megapixels must be between 0.25 and 2")

    return aspect_ratio, megapixels


# --------------------------------------------------
# RunPod handler
# --------------------------------------------------

def handler(job):
    data = job.get("input") or {}
    mode = data.get("mode", "generate")

    if mode not in ("generate", "edit"):
        return {"error": "mode must be 'generate' or 'edit'"}

    prompt = data.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Please provide a non-empty prompt"}

    try:
        steps = int(data.get("num_inference_steps", data.get("steps", 28)))
        guidance_scale = float(data.get("guidance_scale", 4.0))
        num_images = int(data.get("num_images", 1))
        output_format = str(data.get("output_format", "png")).lower()
        seed_value = data.get("seed")
        seed = (
            random.randint(0, 2**64 - 1)
            if seed_value is None
            else int(seed_value)
        )
        aspect_ratio, megapixels = get_image_size(data)

    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}

    if not 1 <= steps <= 60:
        return {"error": "num_inference_steps must be between 1 and 60"}

    if not 0 <= seed <= 2**64 - 1:
        return {"error": "seed must be between 0 and 2^64-1"}

    if not 0 <= guidance_scale <= 30:
        return {"error": "guidance_scale must be between 0 and 30"}

    # Batch generation consumes more VRAM. Start conservatively.
    if not 1 <= num_images <= 4:
        return {"error": "num_images must be between 1 and 4"}

    # PNG is the only format confirmed in the supplied workflow.
    if output_format != "png":
        return {
            "error": (
                "Only output_format='png' is currently supported "
                "by this handler configuration"
            )
        }

    try:
        # ------------------------------
        # Select workflow
        # ------------------------------
        if mode == "generate":
            workflow = load_workflow(WORKFLOW_PATH)

            if PROMPT_NODE not in workflow:
                return {
                    "error": f"Generate prompt node '{PROMPT_NODE}' "
                             "was not found in workflow_api.json"
                }

            workflow[PROMPT_NODE]["inputs"]["prompt"] = prompt

        else:
            workflow = load_workflow(EDIT_WORKFLOW_PATH)

            image_urls = data.get("image_urls")
            old_base64_image = data.get("image")

            # Prefer image_urls; retain compatibility with old Base64 API.
            if image_urls is None:
                if old_base64_image:
                    image_urls = [old_base64_image]
                    input_is_base64 = True
                else:
                    return {
                        "error": (
                            "Edit mode requires image_urls with at least "
                            "one image URL"
                        )
                    }
            else:
                input_is_base64 = False

            if not isinstance(image_urls, list) or not 1 <= len(image_urls) <= 2:
                return {
                    "error": (
                        "image_urls must be a list containing one or "
                        "two images"
                    )
                }

            if EDIT_PROMPT_NODE not in workflow:
                return {
                    "error": f"Edit prompt node '{EDIT_PROMPT_NODE}' "
                             "was not found in workflow_edit_api.json"
                }

            if IMAGE_1_NODE not in workflow:
                return {
                    "error": f"Image node '{IMAGE_1_NODE}' "
                             "was not found in the edit workflow"
                }

            if input_is_base64:
                image_1_name = upload_base64_image_to_comfy(
                    image_urls[0], 1
                )
            else:
                image_1_name = download_image_url(image_urls[0], 1)

            workflow[IMAGE_1_NODE]["inputs"]["image"] = image_1_name
            workflow[EDIT_PROMPT_NODE]["inputs"]["prompt"] = prompt

            # The supplied clothing-transfer workflow has a second
            # LoadImage node for image_2. Use it when a second URL is sent.
            if len(image_urls) == 2:
                if IMAGE_2_NODE not in workflow:
                    return {
                        "error": (
                            "The edit workflow has no second image node "
                            f"'{IMAGE_2_NODE}'"
                        )
                    }

                if input_is_base64:
                    image_2_name = upload_base64_image_to_comfy(
                        image_urls[1], 2
                    )
                else:
                    image_2_name = download_image_url(image_urls[1], 2)

                workflow[IMAGE_2_NODE]["inputs"]["image"] = image_2_name

            # This supplied edit workflow has a switch between the
            # empty latent and the encoded/edit latent paths.
            if LATENT_SWITCH_NODE in workflow:
                workflow[LATENT_SWITCH_NODE]["inputs"]["switch"] = False

        # ------------------------------
        # Shared sampler/resolution settings
        # ------------------------------
        if SAMPLER_NODE not in workflow:
            return {"error": f"Sampler node '{SAMPLER_NODE}' not found"}

        workflow[SAMPLER_NODE]["inputs"]["seed"] = seed
        workflow[SAMPLER_NODE]["inputs"]["steps"] = steps
        workflow[SAMPLER_NODE]["inputs"]["cfg"] = guidance_scale

        if RESOLUTION_NODE in workflow:
            workflow[RESOLUTION_NODE]["inputs"]["aspect_ratio"] = aspect_ratio
            workflow[RESOLUTION_NODE]["inputs"]["megapixels"] = megapixels

        # Configure batch size if this workflow uses EmptyLatentImage.
        if EMPTY_LATENT_NODE in workflow:
            workflow[EMPTY_LATENT_NODE]["inputs"]["batch_size"] = num_images

        # Set output format on SaveImageAdvanced.
        if SAVE_IMAGE_NODE in workflow:
            workflow[SAVE_IMAGE_NODE]["inputs"]["format"] = output_format

        # ------------------------------
        # Submit to ComfyUI
        # ------------------------------
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
            return {
                "error": "ComfyUI did not return a prompt_id",
                "details": response.text[:2000],
            }

        result = wait_for_image(prompt_id)

        if "error" in result:
            return result

        image_bytes = result.pop("image_bytes")
        filename = result["filename"]
        mime_type = result.get("mime_type", "image/png")

        job_id = job.get("id") or str(uuid.uuid4())
        s3_key = upload_result_to_s3(
            image_bytes=image_bytes,
            job_id=job_id,
            filename=filename,
            mime_type=mime_type,
        )

        return {
            "status": "COMPLETED",
            "mode": mode,
            "prompt": prompt,
            "filename": filename,
            "mime_type": mime_type,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "image_size": data.get("image_size", "square"),
            "output_format": output_format,
            "s3_bucket": S3_BUCKET,
            "s3_key": s3_key,
            "prompt_id": prompt_id,
        }

    except requests.RequestException as exc:
        print(f"HTTP/request error: {exc}", flush=True)
        return {
            "error": "Could not communicate with ComfyUI or download an image",
            "details": str(exc),
        }

    except Exception as exc:
        print(f"Unexpected handler error: {exc}", flush=True)
        return {
            "error": "Unexpected handler error",
            "details": str(exc),
        }


def configure_model_paths():
    candidates = [
        "/runpod-volume/qwen-models",
        "/workspace/qwen-models",
    ]

    model_root = next(
        (path for path in candidates if os.path.isdir(path)),
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

    config = f"""
qwen_persistent:
  base_path: "{model_root}/"
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  vae: vae/
"""

    config_path = "/opt/ComfyUI/extra_model_paths.yaml"

    with open(config_path, "w", encoding="utf-8") as file:
        file.write(config)

    print(f"Configured Qwen model paths: {model_root}", flush=True)



# --------------------------------------------------
# Start ComfyUI before accepting RunPod jobs
# --------------------------------------------------

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
        # Stop immediately if ComfyUI crashes during startup.
        if comfy_process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited during startup with code "
                f"{comfy_process.returncode}. Check the worker logs."
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
        "Check the worker logs for startup errors."
    )


# Start ComfyUI first, then accept serverless jobs.
comfy_process = start_comfyui()

runpod.serverless.start({"handler": handler})
