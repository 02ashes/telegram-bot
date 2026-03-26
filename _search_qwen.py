"""Search chat for Qwen-specific discussions about deployment, nodes, and auto-prompt."""
import json

CHAT_PATH = r"C:\Users\1\Downloads\Telegram Desktop\ChatExport_2026-03-25\result.json"

KEYWORDS = [
    "qwen", "Qwen3", "Qwen2", "qwen vl", "qwen3vl", "qwen-vl",
    "提示词助手", "prompt assistant", "promptassistant",
    "提示词小助手", "comfyui-qwen", "ComfyUI-QwenVL",
    "prompt_generator", "LLM节点", "LLM node",
    "LLM-VLM", "vllm", "lm studio",
]

data = json.load(open(CHAT_PATH, "r", encoding="utf-8"))
msgs = data.get("messages", [])

results = []
for m in msgs:
    text = ""
    raw = m.get("text", "")
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        for part in raw:
            if isinstance(part, str):
                text += part
            elif isinstance(part, dict):
                text += part.get("text", "")
    text_lower = text.lower()
    for kw in KEYWORDS:
        if kw.lower() in text_lower:
            results.append({
                "date": m.get("date", ""),
                "from": m.get("from", ""),
                "keyword": kw,
                "text": text[:800],
            })
            break

OUT = r"c:\Users\1\Desktop\Новая папка\telegram_bot\_qwen_chat.txt"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"Found {len(results)} Qwen-related messages\n\n")
    for i, r in enumerate(results):
        f.write(f"=== [{i+1}] [{r['date']}] {r['from']} (kw: {r['keyword']}) ===\n")
        f.write(r["text"][:700] + "\n\n")

print(f"Done. {len(results)} messages -> {OUT}")
