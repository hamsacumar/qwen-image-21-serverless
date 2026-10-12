import os
import json
import time
import uuid
import base64
import subprocess

import requests
import runpod

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "/app/workflow_api.json"
MODEL_PATH = "/workspace/qwen-models"
CONFIG_PATH = "/app/extra_model_paths.yaml"

PROMPT_NODE = "459:471"
SAMPLER_NODE = "459:458"
RESOLUTION_NODE = "13"
SAVE_NODE = "461"


def handler(job):
    data = job.get("input", {})
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

    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)

    workflow[PROMPT_NODE]["inputs"]["prompt"] = prompt
    workflow[SAMPLER_NODE]["inputs"]["seed"] = seed
    workflow[SAMPLER_NODE]["inputs"]["steps"] = steps
    workflow[RESOLUTION_NODE]["inputs"]["aspect_ratio"] = aspect_ratio
    workflow[RESOLUTION_NODE]["inputs"]["megapixels"] = megapixels

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=60,
    )
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]
    deadline = time.monotonic() + 900

    while time.monotonic() < deadline:
        response = requests.get(
            f"{COMFY_URL}/history/{prompt_id}", timeout=30
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
                image_response = requests.get(
                    f"{COMFY_URL}/view",
                    params={
                        "filename": info["filename"],
                        "subfolder": info.get("subfolder", ""),
                        "type": info.get("type", "output"),
                    },
                    timeout=120,
                )
                image_response.raise_for_status()

                return {
                    "image_base64": base64.b64encode(
                        image_response.content
                    ).decode("utf-8"),
                    "mime_type": "image/png",
                    "filename": info["filename"],
                    "seed": seed,
                    "steps": steps,
                }

        time.sleep(2)

    return {"error": "Generation timed out"}


if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(f"Models volume not found: {MODEL_PATH}")

with open(CONFIG_PATH, "w") as f:
    f.write(
        "qwen_persistent:\n"
        f"  base_path: {MODEL_PATH}/\n"
        "  diffusion_models: diffusion_models/\n"
        "  text_encoders: text_encoders/\n"
        "  vae: vae/\n"
    )

comfy_process = subprocess.Popen(
    [
        "python", "/opt/ComfyUI/main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--extra-model-paths-config", CONFIG_PATH,
    ],
    cwd="/opt/ComfyUI",
)

for attempt in range(180):
    if comfy_process.poll() is not None:
        raise RuntimeError("ComfyUI exited during startup")

    try:
        response = requests.get(f"{COMFY_URL}/system_stats", timeout=3)
        if response.ok:
            print("ComfyUI is ready")
            break
    except requests.RequestException:
        pass

    time.sleep(2)
else:
    comfy_process.terminate()
    raise RuntimeError("ComfyUI startup timed out")

runpod.serverless.start({"handler": handler})
