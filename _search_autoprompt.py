"""Search chat export for auto-prompt related discussions."""
import json
import sys

CHAT_PATH = r"C:\Users\1\Downloads\Telegram Desktop\ChatExport_2026-03-25\result.json"
KEYWORDS = [
    "auto prompt", "autoprompt", "auto-prompt", "auto_prompt",
    "llm", "gpt", "enhance prompt", "prompt enhance",
    "prompt rewrite", "prompt expand", "prompt improve",
    "auto generate prompt", "translate prompt", "prompt translation",
    "prompt optimizer", "deepseek", "qwen", "ollama",
    "自动提示", "提示词", "自动翻译", "翻译提示",
    "prompt enhancer", "prompt generator", "smart prompt",
    "ai prompt", "prompt augment", "gemini", "claude",
    "prompt engineering", "prompt template", "negative prompt",
    "auto negative", "quality tags", "booru", "danbooru",
    "prompt builder", "style preset", "style template",
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
                "text": text[:400],
            })
            break

print(f"Found {len(results)} relevant messages")
for i, r in enumerate(results[:60]):
    print(f"\n--- [{i+1}] [{r['date']}] {r['from']} (kw: {r['keyword']})")
    print(r["text"][:300])
