"""ComfyUI API client — RunPod Serverless version.

Thin facade that re-exports all generation functions from the generation/ package.
Import from here (e.g. `import comfyui_api; comfyui_api.run_inpaint(...)`) for
backward compatibility. New code can import directly from generation submodules.
"""

# ── RunPod HTTP client ────────────────────────────────────────
from generation.runpod import (  # noqa: F401
    submit_job,
    poll_job,
    cancel_job,
    get_job_status,
    close_session,
)

# ── Post-processing ──────────────────────────────────────────
from generation.postprocess import (  # noqa: F401
    strip_image_metadata as _strip_image_metadata,
    add_silent_audio as _add_silent_audio,
    extract_file_from_output as _extract_file_from_output,
)

# ── LoRA config ──────────────────────────────────────────────
from generation.loras import (  # noqa: F401
    CHARACTER_LORAS,
    detect_character_loras,
)

# ── Inpaint (Flux Fill) ─────────────────────────────────────
from generation.inpaint import (  # noqa: F401
    build_flux_fill_workflow,
    run_inpaint,
)

# ── Klein Edit (depth + canny) ───────────────────────────────
from generation.klein_edit import (  # noqa: F401
    build_flux_klein_edit_workflow,
    run_image_edit,
    DARK_BEAST_MODELS,
    run_image_edit_dark,
)

# ── Klein Default Edit (ReferenceLatent) ─────────────────────
from generation.klein_default import (  # noqa: F401
    build_flux_klein_default_edit_workflow,
    run_image_edit_default,
)

# ── BFS Face Swap ────────────────────────────────────────────
from generation.bfs import (  # noqa: F401
    build_dark_bfs_workflow,
    run_dark_generate_bfs,
)

# ── Dark Generate (Z-Image Turbo text2img) ───────────────────
from generation.dark_generate import (  # noqa: F401
    build_dark_img2img_workflow,
    build_dark_generate_v2_workflow,
    DARK_GENERATE_MODELS,
    run_dark_generate,
)

# ── Kenpechi Video ───────────────────────────────────────────
from generation.kenpechi import (  # noqa: F401
    check_video_status,
    build_kenpechi_svi_workflow,
    submit_kenpechi_video,
)

# ── ULTIMATE V9 ─────────────────────────────────────────────
from generation.v9 import (  # noqa: F401
    build_v9_workflow,
    run_v9_generate,
)

# ── Final T2I ───────────────────────────────────────────────
from generation.t2i import (  # noqa: F401
    build_t2i_workflow,
    run_t2i_generate,
)
