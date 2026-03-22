"""One-time cleanup script: removes orphaned old video code from comfyui_api.py and app.js.
Run this once from the telegram_bot directory, then delete this script.
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def clean_comfyui_api():
    """Remove orphaned WAN I2V code from comfyui_api.py."""
    path = os.path.join(BASE, "comfyui_api.py")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    original = len(lines)

    # Find Block 1: orphaned VIDEO_ACTIONS + build_wan_i2v_workflow body
    # Starts after last "return workflow" in build_dark_bfs_workflow (after _add_skin_enhance_and_grain with ["40", 0])
    # Ends before "# RunPod Serverless API" section
    block1_start = block1_end = None
    for i, line in enumerate(lines):
        if '_add_skin_enhance_and_grain(workflow, ["40", 0], "61")' in line:
            # Next non-empty line should be "return workflow", then blank lines
            block1_start = i + 2  # index of "return workflow" line
            # Skip the return and any blanks to find start of orphan
            j = block1_start + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            block1_start = block1_start + 1  # keep "return workflow", delete from next line
            break

    for i, line in enumerate(lines):
        if '# RunPod Serverless API' in line:
            block1_end = i - 1  # line before the === header
            break

    # Find Block 2: orphaned submit_video body
    # Starts after run_dark_generate's "return None" (after "Failed to parse dark generate result")
    # Ends before "def _add_silent_audio"
    block2_start = block2_end = None
    for i, line in enumerate(lines):
        if 'Failed to parse dark generate result' in line:
            # Pattern: logger.error line, "return None", blank, blank, orphan starts
            block2_start = i + 2  # after "return None"
            j = block2_start + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            block2_start = block2_start + 1  # keep "return None", start deleting from next line
            break

    for i, line in enumerate(lines):
        if 'def _add_silent_audio(video_bytes: bytes)' in line:
            block2_end = i  # delete up to but not including this line
            break

    print(f"comfyui_api.py: {original} lines")
    if block1_start and block1_end and block1_start < block1_end:
        print(f"  Block 1 (WAN I2V): lines {block1_start+1}-{block1_end+1} ({block1_end - block1_start + 1} lines)")
    if block2_start and block2_end and block2_start < block2_end:
        print(f"  Block 2 (submit_video): lines {block2_start+1}-{block2_end+1} ({block2_end - block2_start + 1} lines)")

    # Remove block 2 first (preserves block 1 indices)
    new_lines = []
    for i, line in enumerate(lines):
        in_block1 = block1_start and block1_end and block1_start <= i <= block1_end
        in_block2 = block2_start and block2_end and block2_start <= i < block2_end
        if in_block1 or in_block2:
            continue
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"  Result: {len(new_lines)} lines (removed {original - len(new_lines)})")


def clean_app_js():
    """Remove old generateVideo function and dead _originalGenerateVideo from app.js."""
    path = os.path.join(BASE, "webapp", "app.js")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    original = len(lines)

    # Block 1: old generateVideo function (from "async function generateVideo(prompt)" to "}")
    # It's between "// Video Generation" section and "// Image Edit Generation" section
    block1_start = block1_end = None
    for i, line in enumerate(lines):
        if 'async function generateVideo(prompt) {' in line:
            # Include the comment headers above
            j = i - 1
            while j >= 0 and (lines[j].strip().startswith('//') or lines[j].strip() == ''):
                if '// Video Generation' in lines[j]:
                    break
                j -= 1
            block1_start = j  # start from "// Video Generation" comment
            break

    for i, line in enumerate(lines):
        if '// Image Edit Generation (Flux 2 Klein)' in line:
            # Previous line should be section comment start
            j = i - 1
            while j >= 0 and lines[j].strip() == '':
                j -= 1
            block1_end = j + 1  # include trailing blank lines
            break

    # Block 2: dead _originalGenerateVideo line + comment
    block2_lines = []
    for i, line in enumerate(lines):
        if '_originalGenerateVideo' in line:
            # Also remove the comment above it
            if i > 0 and 'Override generateVideo' in lines[i-1]:
                block2_lines.extend([i-1, i])
            else:
                block2_lines.append(i)

    # Also remove monkey-patch comment at end of file
    for i, line in enumerate(lines):
        if "Monkey-patch: if mode is 'video', use Kenpechi instead of old generateVideo" in line:
            block2_lines.append(i)
        if '_origPrepareAndGenerate' in line:
            block2_lines.append(i)

    print(f"app.js: {original} lines")
    if block1_start is not None and block1_end is not None:
        print(f"  Block 1 (old generateVideo): lines {block1_start+1}-{block1_end+1} ({block1_end - block1_start + 1} lines)")
    if block2_lines:
        print(f"  Block 2 (dead refs): lines {[l+1 for l in block2_lines]}")

    # Build output
    skip_set = set(block2_lines)
    new_lines = []
    for i, line in enumerate(lines):
        in_block1 = block1_start is not None and block1_end is not None and block1_start <= i <= block1_end
        if in_block1 or i in skip_set:
            continue
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"  Result: {len(new_lines)} lines (removed {original - len(new_lines)})")


if __name__ == "__main__":
    clean_comfyui_api()
    print()
    clean_app_js()
    print("\n✅ Done! You can delete _cleanup.py now.")
