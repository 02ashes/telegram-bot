import json

with open(r'C:\Users\1\Downloads\Telegram Desktop\ChatExport_2026-02-22\result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def get_text(m):
    text = m.get('text', '')
    if isinstance(text, list):
        parts = []
        for p in text:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get('text', ''))
        return ''.join(parts)
    return text if isinstance(text, str) else ''

keywords = ['训练', 'train', 'rank', 'adapter', 'toolkit', '步数', 'step', 'learning', '底模', 'base model', 'diffuser', 'kohya', 'caption', '打标', '权重', 'weight', 'alpha', '练lora', '炼lora', 'ostris', '人脸', 'face', '换脸']

out = []
for m in data['messages']:
    text = get_text(m)
    text_lower = text.lower()
    if 'lora' in text_lower and any(kw in text_lower for kw in keywords):
        text_clean = text.replace('\n', ' | ').strip()
        if len(text_clean) > 500:
            text_clean = text_clean[:500] + '...'
        out.append(f"[{m.get('date','?')}] {m.get('from','?')}: {text_clean}\n")

with open(r'C:\Users\1\Documents\ComfyUI\telegram_bot\_lora_messages.txt', 'w', encoding='utf-8') as f:
    f.write(f"Found {len(out)} messages\n\n")
    f.writelines(out)

print(f"Done! {len(out)} messages saved to _lora_messages.txt")
