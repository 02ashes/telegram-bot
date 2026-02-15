"""RunPod pod manager — auto start/stop on demand."""

import asyncio
import logging
import time

import aiohttp
import runpod

import config

logger = logging.getLogger(__name__)

runpod.api_key = config.RUNPOD_API_KEY

_last_activity: float = 0
_idle_task: asyncio.Task | None = None
_pod_running: bool = False


async def _check_comfyui_ready() -> bool:
    """Check if ComfyUI API is responding."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{config.COMFYUI_BASE_URL}/system_stats",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def ensure_pod_running() -> bool:
    """Start the pod if it's not running, wait for ComfyUI to be ready."""
    global _pod_running, _last_activity

    _last_activity = time.time()

    # Quick check — maybe it's already running
    if _pod_running and await _check_comfyui_ready():
        _reset_idle_timer()
        return True

    logger.info("Starting RunPod pod %s...", config.RUNPOD_POD_ID)

    try:
        # Resume the pod (start it from stopped state)
        result = runpod.resume_pod(config.RUNPOD_POD_ID)
        logger.info("resume_pod result: %s", result)
    except Exception as e:
        # Pod might already be running
        logger.warning("resume_pod error (might be already running): %s", e)

    # Wait for ComfyUI to become ready (up to 3 minutes)
    logger.info("Waiting for ComfyUI to start...")
    for i in range(36):  # 36 * 5s = 180s = 3 min
        await asyncio.sleep(5)
        if await _check_comfyui_ready():
            logger.info("ComfyUI is ready! (took ~%d seconds)", (i + 1) * 5)
            _pod_running = True
            _reset_idle_timer()
            return True
        logger.info("  Still waiting... (%d/36)", i + 1)

    logger.error("ComfyUI did not start within 3 minutes!")
    return False


async def stop_pod():
    """Stop the RunPod pod."""
    global _pod_running

    logger.info("Stopping RunPod pod %s...", config.RUNPOD_POD_ID)
    try:
        result = runpod.stop_pod(config.RUNPOD_POD_ID)
        logger.info("stop_pod result: %s", result)
        _pod_running = False
    except Exception as e:
        logger.error("stop_pod error: %s", e)


def _reset_idle_timer():
    """Reset the idle timer — pod will stop after IDLE_TIMEOUT_SECONDS."""
    global _idle_task, _last_activity

    _last_activity = time.time()

    if _idle_task and not _idle_task.done():
        _idle_task.cancel()

    _idle_task = asyncio.create_task(_idle_shutdown())


async def _idle_shutdown():
    """Wait for inactivity, then stop the pod."""
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            elapsed = time.time() - _last_activity
            if elapsed >= config.IDLE_TIMEOUT_SECONDS:
                logger.info(
                    "Pod idle for %d seconds, stopping...",
                    int(elapsed),
                )
                await stop_pod()
                return
    except asyncio.CancelledError:
        pass


def touch_activity():
    """Mark activity to prevent idle shutdown."""
    global _last_activity
    _last_activity = time.time()
