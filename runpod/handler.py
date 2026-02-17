"""
Patched handler.py for RunPod Serverless ComfyUI Worker.

Based on runpod-workers/worker-comfyui handler.py (5.5.1).
PATCH: Also collects "gifs" output from VHS_VideoCombine nodes,
       which outputs video under that key instead of "images".
"""

import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback
import logging

try:
    from network_volume import (
        is_network_volume_debug_enabled,
        run_network_volume_diagnostics,
    )
except ImportError:
    def is_network_volume_debug_enabled():
        return False
    def run_network_volume_diagnostics():
        pass

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMFY_API_AVAILABLE_INTERVAL_MS = 50
COMFY_API_AVAILABLE_MAX_RETRIES = 500
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)

COMFY_HOST = "127.0.0.1:8188"
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


# ---------------------------------------------------------------------------
def _comfy_server_status():
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    print(f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect...")
    last_reconnect_error = initial_error
    for attempt in range(max_attempts):
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            print(f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect")
            raise websocket.WebSocketConnectionClosedException("ComfyUI HTTP unreachable during websocket reconnect")

        print(f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}...")
        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print(f"worker-comfyui - Websocket reconnected successfully.")
            return new_ws
        except (websocket.WebSocketException, ConnectionRefusedError, socket.timeout, OSError) as reconn_err:
            last_reconnect_error = reconn_err
            print(f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}")
            if attempt < max_attempts - 1:
                time.sleep(delay_s)

    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


def validate_input(job_input):
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return None, "'images' must be a list of objects with 'name' and 'image' keys"

    comfy_org_api_key = job_input.get("comfy_org_api_key")

    return {
        "workflow": workflow,
        "images": images,
        "comfy_org_api_key": comfy_org_api_key,
    }, None


def check_server(url, retries=500, delay=50):
    print(f"worker-comfyui - Checking API server at {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                return True
        except requests.RequestException:
            pass
        time.sleep(delay / 1000)

    print(f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts.")
    return False


def upload_images(images):
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")
    for image in images:
        try:
            name = image["name"]
            image_data_uri = image["image"]

            if "," in image_data_uri:
                base64_data = image_data_uri.split(",", 1)[1]
            else:
                base64_data = image_data_uri

            blob = base64.b64decode(base64_data)
            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }
            response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=30)
            response.raise_for_status()
            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")
        except Exception as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        return {"status": "error", "message": "Some images failed to upload", "details": upload_errors}

    return {"status": "success", "message": "All images uploaded successfully", "details": responses}


def get_available_models():
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()
        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )
        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    payload = {"prompt": workflow, "client_id": client_id}
    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30)

    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = response.json()
            error_message = "Workflow validation failed"
            error_details = []

            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                else:
                    error_message = str(error_info)

            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(f"Node {node_id} ({error_type}): {error_msg}")
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            if error_data.get("type") == "prompt_outputs_failed_validation":
                available_models = get_available_models()
                if available_models.get("checkpoints"):
                    error_message += f"\n\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"

            if error_details:
                detailed_message = f"{error_message}:\n" + "\n".join(f"• {detail}" for detail in error_details)
                if any("not in list" in detail and "ckpt_name" in detail for detail in error_details):
                    available_models = get_available_models()
                    if available_models.get("checkpoints"):
                        detailed_message += f"\n\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                raise ValueError(detailed_message)
            else:
                raise ValueError(f"{error_message}. Raw response: {response.text}")
        except (json.JSONDecodeError, KeyError):
            raise ValueError(f"ComfyUI validation failed: {response.text}")

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_image_data(filename, subfolder, image_type):
    """Fetch file bytes (image or video) from the ComfyUI /view endpoint."""
    print(f"worker-comfyui - Fetching file data: type={image_type}, subfolder={subfolder}, filename={filename}")
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    try:
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=120)
        response.raise_for_status()
        print(f"worker-comfyui - Successfully fetched data for {filename} ({len(response.content)} bytes)")
        return response.content
    except Exception as e:
        print(f"worker-comfyui - Error fetching data for {filename}: {e}")
        return None


def _process_output_items(items, job_id, output_data, errors):
    """
    Process a list of output file items (images or video/gifs).
    Each item is a dict with {filename, subfolder, type}.
    """
    for file_info in items:
        filename = file_info.get("filename")
        subfolder = file_info.get("subfolder", "")
        file_type = file_info.get("type")

        # Skip temp files
        if file_type == "temp":
            print(f"worker-comfyui - Skipping {filename} (type=temp)")
            continue

        if not filename:
            errors.append(f"Skipping item due to missing filename: {file_info}")
            continue

        file_bytes = get_image_data(filename, subfolder, file_type)
        if not file_bytes:
            errors.append(f"Failed to fetch data for {filename}")
            continue

        file_extension = os.path.splitext(filename)[1] or ".png"

        if os.environ.get("BUCKET_ENDPOINT_URL"):
            try:
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                    temp_file.write(file_bytes)
                    temp_file_path = temp_file.name

                print(f"worker-comfyui - Uploading {filename} to S3...")
                s3_url = rp_upload.upload_image(job_id, temp_file_path)
                os.remove(temp_file_path)
                print(f"worker-comfyui - Uploaded {filename} to S3: {s3_url}")
                output_data.append({"filename": filename, "type": "s3_url", "data": s3_url})
            except Exception as e:
                errors.append(f"Error uploading {filename} to S3: {e}")
        else:
            try:
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                output_data.append({"filename": filename, "type": "base64", "data": base64_data})
                print(f"worker-comfyui - Encoded {filename} as base64 ({len(base64_data)} chars)")
            except Exception as e:
                errors.append(f"Error encoding {filename} to base64: {e}")


# ---------------------------------------------------------------------------
# MAIN HANDLER
# ---------------------------------------------------------------------------
def handler(job):
    if is_network_volume_debug_enabled():
        run_network_volume_diagnostics()

    job_input = job["input"]
    job_id = job["id"]

    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")

    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {"error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."}

    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            return {"error": "Failed to upload input images", "details": upload_result["details"]}

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    errors = []

    try:
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-comfyui - Websocket connected")

        try:
            queued_workflow = queue_workflow(
                workflow, client_id,
                comfy_org_api_key=validated_data.get("comfy_org_api_key"),
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"Missing 'prompt_id' in queue response: {queued_workflow}")
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Error queuing workflow: {e}")

        # Wait for execution completion via WebSocket
        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...")
        execution_done = False
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        status_data = message.get("data", {}).get("status", {})
                        print(f"worker-comfyui - Queue remaining: {status_data.get('exec_info', {}).get('queue_remaining', 'N/A')}")
                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            print(f"worker-comfyui - Execution finished for prompt {prompt_id}")
                            execution_done = True
                            break
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, Message: {data.get('exception_message')}"
                            errors.append(f"Workflow execution error: {error_details}")
                            break
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    ws = _attempt_websocket_reconnect(
                        ws_url, WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S, closed_err,
                    )
                    continue
                except websocket.WebSocketConnectionClosedException as reconn_failed_err:
                    raise reconn_failed_err
            except json.JSONDecodeError:
                print(f"worker-comfyui - Received invalid JSON via websocket.")

        if not execution_done and not errors:
            raise ValueError("Workflow monitoring loop exited without completion or error.")

        # Fetch history
        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history."
            if not errors:
                return {"error": error_msg}
            errors.append(error_msg)
            return {"error": "Job processing failed", "details": errors}

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        if not outputs:
            errors.append(f"No outputs found in history for prompt {prompt_id}.")

        # =================================================================
        # PATCHED: Process BOTH "images" AND "gifs" output keys.
        #
        # Standard ComfyUI nodes (SaveImage, SaveAnimatedWEBP, etc.)
        # put output under the "images" key.
        #
        # VHS_VideoCombine puts video output under the "gifs" key.
        # The stock handler ignores "gifs" → videos are lost.
        # This patch collects both.
        # =================================================================
        print(f"worker-comfyui - Processing {len(outputs)} output nodes...")
        for node_id, node_output in outputs.items():
            # Process standard image output
            if "images" in node_output:
                print(f"worker-comfyui - Node {node_id}: {len(node_output['images'])} image(s)")
                _process_output_items(node_output["images"], job_id, output_data, errors)

            # PATCH: Process VHS_VideoCombine "gifs" output (video files)
            if "gifs" in node_output:
                print(f"worker-comfyui - Node {node_id}: {len(node_output['gifs'])} video/gif file(s) [VHS output]")
                _process_output_items(node_output["gifs"], job_id, output_data, errors)

            # Log any other unhandled keys
            handled_keys = {"images", "gifs"}
            other_keys = [k for k in node_output.keys() if k not in handled_keys]
            if other_keys:
                print(f"worker-comfyui - WARNING: Node {node_id} has unhandled output keys: {other_keys}")

    except websocket.WebSocketException as e:
        print(f"worker-comfyui - WebSocket Error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}
    except requests.RequestException as e:
        print(f"worker-comfyui - HTTP Request Error: {e}")
        print(traceback.format_exc())
        return {"error": f"HTTP communication error with ComfyUI: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - Value Error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    except Exception as e:
        print(f"worker-comfyui - Unexpected Handler Error: {e}")
        print(traceback.format_exc())
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if ws and ws.connected:
            ws.close()

    final_result = {}
    if output_data:
        final_result["images"] = output_data
    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - Job completed with errors: {errors}")

    if not output_data and errors:
        return {"error": "Job processing failed", "details": errors}
    elif not output_data and not errors:
        final_result["status"] = "success_no_images"
        final_result["images"] = []

    print(f"worker-comfyui - Job completed. Returning {len(output_data)} file(s).")

    if REFRESH_WORKER:
        final_result["refresh_worker"] = True

    return final_result


if __name__ == "__main__":
    print("worker-comfyui - Starting patched handler (with VHS video support)...")
    runpod.serverless.start({"handler": handler})
