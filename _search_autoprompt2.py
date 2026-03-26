"""Search chat export for detailed auto-prompt implementation discussions - outputs to file."""
import json

CHAT_PATH = r"C:\Users\1\Downloads\Telegram Desktop\ChatExport_2026-03-25\result.json"

# Implementation-level keywords
KEYWORDS = [
    "autoprompt", "auto-prompt", "auto prompt",
    "prompt enhance", "prompt rewrite", "prompt expand",
    "prompt optimizer", "prompt builder",
    "smart prompt", "prompt generator",
    "style preset", "style template",
    "booru tag", "danbooru", "quality tags",
    "prompt template", "negative prompt",
    "auto negative",
    "florence", "joycaption", "joy caption",
    "caption model", "reverse prompt",
    "prompt api", "prompt node",
    "ollama", "gemini api", "openai api",
    "deepseek",
]

# Chinese keywords
CN_KEYWORDS = [
    "LLM", "描述词", "反推",
    "提示词增强", "提示词优化", "提示词模板",
    "自动提示", "自动翻译",
    "文生图提示词专家",
]

ALL_KEYWORDS = KEYWORDS + CN_KEYWORDS

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
    for kw in ALL_KEYWORDS:
        if kw.lower() in text_lower:
            results.append({
                "date": m.get("date", ""),
                "from": m.get("from", ""),
                "keyword": kw,
                "text": text[:600],
            })
            break

OUT_PATH = r"c:\Users\1\Desktop\Новая папка\telegram_bot\chat_summary.txt"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Found {len(results)} relevant messages\n\n")
    for i, r in enumerate(results):
        f.write(f"=== [{i+1}] [{r['date']}] {r['from']} (kw: {r['keyword']}) ===\n")
        f.write(r["text"][:500] + "\n\n")

print(f"Done. Found {len(results)} messages. Written to {OUT_PATH}")
