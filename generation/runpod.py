"""RunPod Serverless HTTP client — submit, poll, cancel, status."""

import asyncio
import logging

import aiohttp

import config

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# Global aiohttp session — reused across all RunPod API calls
_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    """Get or create a global aiohttp session for connection reuse."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {config.RUNPOD_API_KEY}"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
    return _session


async def close_session():
    """Close the global aiohttp session (call at shutdown)."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def submit_job(workflow: dict, images: list[dict], endpoint_id: str | None = None) -> str | None:
    """Submit a job to RunPod Serverless endpoint.
    
    Args:
        workflow: ComfyUI workflow dict
        images: List of {"name": "filename.png", "image": "base64data"}
        endpoint_id: RunPod endpoint ID (defaults to config.RUNPOD_ENDPOINT_ID)
    
    Returns:
        Job ID or None on failure.
    """
    ep = endpoint_id or config.RUNPOD_ENDPOINT_ID
    url = f"{RUNPOD_API_BASE}/{ep}/run"
    payload = {
        "input": {
            "workflow": workflow,
            "images": images,
        }
    }
    
    session = _get_session()
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            logger.error("RunPod submit failed (%d): %s", resp.status, text)
            return None
        result = await resp.json()
        job_id = result.get("id")
        logger.info("RunPod job submitted: %s (status: %s)", job_id, result.get("status"))
        return job_id


async def poll_job(job_id: str, timeout: int = 1200, endpoint_id: str | None = None) -> dict | None:
    """Poll RunPod Serverless for job completion.
    
    Returns the output dict or None on timeout/failure.
    Default timeout: 1200s (20 minutes) to handle long NSFW video generation.
    """
    ep = endpoint_id or config.RUNPOD_ENDPOINT_ID
    url = f"{RUNPOD_API_BASE}/{ep}/status/{job_id}"
    session = _get_session()
    
    for i in range(timeout // 3):
        await asyncio.sleep(3)
        try:
            async with session.get(url) as resp:
                data = await resp.json()
                status = data.get("status")
                
                if status == "COMPLETED":
                    logger.info("Job %s completed!", job_id)
                    return data.get("output")
                elif status == "FAILED":
                    logger.error("Job %s failed: %s", job_id, data.get("error"))
                    return None
                elif status == "CANCELLED":
                    logger.info("Job %s was cancelled", job_id)
                    return None
                elif status in ("IN_QUEUE", "IN_PROGRESS"):
                    if i % 5 == 0:
                        logger.info("Job %s status: %s", job_id, status)
                else:
                    logger.warning("Job %s unknown status: %s", job_id, status)
        except Exception as e:
            logger.warning("Poll error: %s", e)

    logger.error("Job %s timed out after %d seconds", job_id, timeout)
    return None


async def cancel_job(job_id: str, endpoint_id: str | None = None) -> bool:
    """Cancel a running or queued RunPod job.
    
    Returns True if cancel request was sent successfully.
    """
    ep = endpoint_id or config.RUNPOD_ENDPOINT_ID
    url = f"{RUNPOD_API_BASE}/{ep}/cancel/{job_id}"
    
    try:
        session = _get_session()
        async with session.post(url) as resp:
            if resp.status == 200:
                logger.info("Job %s cancel requested", job_id)
                return True
            else:
                logger.warning("Failed to cancel job %s: %s", job_id, resp.status)
                return False
    except Exception as e:
        logger.error("Cancel error: %s", e)
        return False


async def get_job_status(job_id: str, endpoint_id: str | None = None) -> dict:
    """Get current status of a RunPod job.
    
    Returns dict with {status, output?}.
    """
    ep = endpoint_id or config.RUNPOD_ENDPOINT_ID
    url = f"{RUNPOD_API_BASE}/{ep}/status/{job_id}"
    
    try:
        session = _get_session()
        async with session.get(url) as resp:
            data = await resp.json()
            return {
                "status": data.get("status", "UNKNOWN"),
                "output": data.get("output"),
                "error": data.get("error"),
            }
    except Exception as e:
        logger.error("Status check error: %s", e)
        return {"status": "ERROR", "error": str(e)}
