---
description: How to train Face LoRA for Flux 2 Klein 9B on RunPod
---

# 🎓 Face LoRA Training — Полный Гайд
## Flux 2 Klein 9B | ai-toolkit | RunPod

> Источники: официальная документация ai-toolkit, Reddit, CivitAI, ZImage.run,
> китайское сообщество (AiMetatron GY, BMX 726, feishu.cn, smzdm.com),
> runcomfy.com, HuggingFace, arXiv.

---

## 📸 Часть 1: Подготовка датасета (КРИТИЧЕСКИ ВАЖНО)

### 1.1 Сколько фоток нужно

| Количество | Вердикт |
|---|---|
| 5–10 | Минимум. Может работать, но нестабильно |
| **15–25** | **✅ Идеально для лица. Золотая середина** |
| 25–40 | Хорошо если есть разнообразие. Больше гибкости |
| 50+ | ⚠️ Риск переобучения. Diminishing returns |

> **Правило из китайского сообщества**: "写实人物图片 30-50 次 repeat" — 
> для реалистичных людей нужно 30-50 repeats на каждое фото.
> При 20 фотках × 30 repeats = 600 шагов на эпоху.

### 1.2 Какие фотки включать

```
✅ ОБЯЗАТЕЛЬНО (разнообразие = качество):
├── 5-7 крупный план лица (разные выражения)
│   ├── улыбка
│   ├── серьёзное лицо  
│   ├── рот открыт / закрыт
│   └── с макияжем / без
├── 3-4 портрет 3/4 ракурс (слева + справа)
├── 2-3 профиль (вид сбоку)
├── 3-4 по грудь / по пояс (разная одежда!)
├── 2-3 полный рост (стоя, сидя)
└── 2-3 NSFW (если нужно чтобы LoRA знала тело)

❌ УБРАТЬ (испортят результат):
├── Лицо закрыто / не видно / размыто
├── Маски, тёмные солнечные очки
├── Фото только тела без лица
├── Несколько людей на одном фото
├── Водяные знаки / текст
├── Экстремальные ракурсы (снизу/сверху сильно)
└── Одинаковые фото с одного ракурса (дубли)
```

### 1.3 Требования к фоткам

| Параметр | Требование |
|---|---|
| **Разрешение** | Минимум 512×512, лучше 1024×1024+ |
| **Формат** | JPG, PNG, WebP |
| **Соотношение** | Любое (ai-toolkit сам подгонит) |
| **Фон** | РАЗНЫЙ! Не все на одном фоне |
| **Освещение** | Разное (дневной свет, студия, тёмное) |
| **Одежда** | Разная! Иначе LoRA "выучит" одну футболку |
| **Качество** | Чёткое, не размытое, без артефактов сжатия |

> ⚠️ **ВАЖНО из ZImage community**: "Если все фотки на чёрном фоне — LoRA привяжется 
> к чёрному фону. Если все в одной футболке — LoRA привяжется к футболке."

### 1.4 Captioning (описания) — САМАЯ ВАЖНАЯ ЧАСТЬ

Для КАЖДОЙ фотки `XXX.jpg` нужен файл `XXX.txt` с описанием.

#### Правило captioning:

> **Описывай то, что ты хочешь менять. НЕ описывай то, что определяет личность.**

Это значит:
- ✅ Описывай: одежду, позу, фон, освещение, выражение лица
- ❌ НЕ описывай: "girl with blue eyes and button nose" — это LoRA выучит сама

#### Формат caption:

```
[trigger_word], [описание того что ПЕРЕМЕННОЕ на фото]
```

#### Примеры хороших captions:

```
misu, a young woman taking a selfie, wearing black lingerie, dark bedroom, warm lamp lighting, looking at camera with playful expression

misu, a woman in white fur coat, red lipstick, studio photography, neutral background, soft diffused lighting, portrait shot

misu, a girl sitting on bed, topless, casual pose, messy hair, natural window lighting, bedroom background

misu, a woman standing in front of mirror, cat ear headband, black outfit, full body shot, dim ambient lighting
```

#### Примеры ПЛОХИХ captions:

```
❌ misu  
   (слишком короткий — нет контекста)

❌ a beautiful girl with long black hair, glasses, pale skin, small nose
   (описывает ЛИЦО — LoRA должна выучить это сама)

❌ photo of a person
   (слишком общий — бесполезен)
```

#### Про trigger word:

| Правило | Описание |
|---|---|
| **Уникальное слово** | НЕ "girl", НЕ "woman", НЕ "cute". Используй имя: `misu`, `jane`, `anya` |
| **В начале caption** | Всегда первое слово |
| **Одно и то же** | Одинаковое во всех .txt файлах одной модели |
| **Без пробелов** | `misu` ✅, `misu model` ❌ |

### 1.5 Авто-captioning

// turbo
Запусти мой скрипт `auto_caption.py`:
```bash
pip install transformers torch pillow
python auto_caption.py --dir "D:\AI TRAIN\Misu" --trigger "misu"
```

Это создаст `.txt` файлы автоматически с помощью BLIP, но **ОБЯЗАТЕЛЬНО проверь и подправь** каждый caption вручную! Автоматические описания часто неточные.

### 1.6 Финальная структура папки

```
D:\AI TRAIN\Misu\
├── Misu1.jpg
├── Misu1.txt    → "misu, a young woman selfie, black lingerie, dark room..."
├── Misu2.jpg
├── Misu2.txt    → "misu, a woman portrait, glasses, white top, natural light..."
├── Misu3.jpg
├── Misu3.txt    → "misu, a girl full body, standing, casual outfit, outdoor..."
├── ...
└── Misu20.jpg
    Misu20.txt   → "misu, a woman close-up face, red lipstick, studio light..."
```

---

## ⚙️ Часть 2: Параметры тренировки (ВСЕ детали)

### 2.1 Основные параметры

| Параметр | Значение | Объяснение |
|---|---|---|
| **Rank (network_dim)** | **64–128** | Сколько "слотов памяти" у LoRA. Для лица нужно 64+. Китайское сообщество рекомендует 128 для DiT моделей |
| **Alpha (network_alpha)** | **= Rank** или **Rank/2** | Масштабный коэффициент. alpha=rank → нейтральный эффект. alpha=rank/2 → более стабильно |
| **Learning Rate** | **1e-4 — 2e-4** | Скорость обучения. 1e-4 для маленьких датасетов (15 фоток), 2e-4 для средних (30+) |
| **LR Scheduler** | **cosine** | Плавно уменьшает LR к концу тренировки. Лучше чем constant |
| **Steps** | **1500–2500** | Для 20 фоток ~2000 steps оптимально. Формула: ~100 steps × кол-во фоток |
| **Batch Size** | **1** | На 16GB VRAM только 1. Но это нормально для face LoRA |
| **Optimizer** | **adamw8bit** | 8-bit AdamW экономит VRAM без потери качества |
| **Resolution** | **1024×1024** | Нативное разрешение Klein. НЕ 512! |
| **Gradient Checkpointing** | **true** | Экономит VRAM ценой ~20% замедления |
| **Mixed Precision** | **bf16** | RTX 5070 Ti поддерживает bf16 нативно |
| **Quantize base model** | **true** | 4-bit квантизация базовой модели. Без этого 16GB не хватит! |

### 2.2 Формула расчёта steps

```
Рекомендуемые steps = кол-во фоток × repeats × epochs

Пример:
  20 фоток × 30 repeats × 3 epochs = 1800 steps

Сохранение каждые 250-500 steps → потом выбираешь лучший чекпоинт
```

### 2.3 Что НЕ тренировать

| Компонент | Тренировать? | Почему |
|---|---|---|
| **UNet/DiT** | ✅ Да | Основная визуальная модель — здесь "живёт" лицо |
| **Text Encoder** | ❌ Нет | Для Flux/Klein text encoder НЕ тренируется. Это не SD 1.5! |

### 2.4 Caption dropout

Установи `caption_dropout_rate: 0.05` — в 5% случаев caption будет пропускаться. 
Это учит LoRA ассоциировать trigger word с лицом даже без описания.

---

## 🚨 Часть 3: Ошибки и как их избежать

### 3.1 Переобучение (Overfitting) — ГЛАВНЫЙ ВРАГ

**Признаки:**
- Все генерации выглядят одинаково
- LoRA "застряла" на одной позе/ракурсе из датасета
- Фон из тренировочных фоток лезет в результат
- Цвета пересыщенные, тёмные тени чёрные, засветы выбитые ("burnt images")
- Нельзя изменить цвет волос или добавить бороду

**Как избежать:**
1. ✅ Разнообразный датасет (разные ракурсы, фоны, одежда)
2. ✅ `save_every: 250` — сохранять чекпоинты и тестировать
3. ✅ LR не выше 2e-4
4. ✅ Не больше 3000 steps для 20 фоток
5. ✅ caption_dropout_rate: 0.05
6. ✅ Cosine LR scheduler

### 3.2 Недообучение (Underfitting)

**Признаки:**
- LoRA почти не влияет на результат
- Лицо не похоже на оригинал
- Trigger word не работает

**Как исправить:**
1. ✅ Увеличь steps (попробуй 2500-3000)
2. ✅ Увеличь LR (попробуй 3e-4)
3. ✅ Увеличь rank (попробуй 128)
4. ✅ Проверь captions — trigger word на месте?

### 3.3 Частые ошибки новичков

| Ошибка | Последствие | Решение |
|---|---|---|
| Все фотки на одном фоне | LoRA привязывается к фону | Разные фоны |
| Описал лицо в caption | LoRA не учит лицо как "identity" | Не описывай лицо |
| Rank слишком маленький (8-16) | Не хватает "памяти" для лица | Минимум 64, лучше 128 |
| Слишком много steps | Переобучение | 1500-2500 для 20 фоток |
| Не проверял чекпоинты | Не знаешь какой лучший | save_every: 250 |
| Забыл trigger word в caption | LoRA не привязана к слову | Всегда в начале |
| Фотки low-res (256px) | Модель учит размытые лица | Минимум 512, лучше 1024 |
| Klein 9B crash при тренировке | Нет квантизации | quantize: true |

---

## 🖥️ Часть 4: RunPod Setup (пошагово)

### 4.1 Арендовать GPU

1. Зайди на [runpod.io](https://runpod.io)
2. **GPU Pods** → **Deploy**
3. Выбери GPU:

| GPU | VRAM | Цена/час | Для кого |
|---|---|---|---|
| **A6000 48GB** | 48 GB | ~$0.76 | ✅ Бюджетный вариант |
| **A100 80GB** | 80 GB | ~$1.64 | ✅ Быстрее и стабильнее |
| **H100 80GB** | 80 GB | ~$3.89 | Overkill, не нужно |

4. Template: **RunPod PyTorch 2.4** (или любой с CUDA 12+)
5. Volume Disk: **50 GB** минимум
6. Нажми **Deploy On-Demand**

### 4.2 Установка ai-toolkit

Когда pod запустится → **Connect** → **Terminal**:

```bash
# 1. Клонировать ai-toolkit
cd /workspace
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
git submodule update --init --recursive

# 2. Установить зависимости
pip install -r requirements.txt
pip install peft bitsandbytes accelerate

# 3. Подготовить папки
mkdir -p /workspace/models
mkdir -p /workspace/datasets
mkdir -p /workspace/output
```

### 4.3 Загрузить базовую модель

Нужен файл `flux-2-klein-9b.safetensors`. Варианты:

**Вариант A — с HuggingFace:**
```bash
pip install huggingface_hub
huggingface-cli download black-forest-labs/FLUX.2-klein \
    flux-2-klein-9b.safetensors \
    --local-dir /workspace/models/
```

**Вариант B — скопировать с RunPod volume (если уже есть):**
```bash
cp /runpod-volume/ComfyUI/models/unet/flux-2-klein-9b.safetensors /workspace/models/
```

### 4.4 Загрузить датасет

Через **RunPod File Manager** (Upload) или через `scp`:

```bash
# Локально на Windows (в PowerShell):
scp -P PORT -r "D:\AI TRAIN\Misu\*" root@POD_IP:/workspace/datasets/misu/
```

Или загрузи через веб-интерфейс RunPod.

### 4.5 Создать конфиг тренировки

Создай файл `/workspace/ai-toolkit/config/train_misu.yaml`:

```yaml
job: extension
config:
  name: "misu_face_lora"
  process:
    - type: "sd_trainer"
      training_folder: "/workspace/output"
      device: "cuda:0"
      trigger_word: "misu"
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
      save:
        dtype: fp16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/workspace/datasets/misu"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [1024, 1024]
          default_caption: "misu"
          cache_latents_to_disk: true
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1.5e-4
        lr_scheduler: "cosine"
        max_denoising_steps: 50
        dtype: bf16
      model:
        name_or_path: "/workspace/models/flux-2-klein-9b.safetensors"
        is_flux: true
        quantize: true
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
          - "misu, a woman portrait, studio lighting, neutral background"
          - "misu, a woman sitting on a sofa, casual outfit, natural light"
          - "misu, a girl close-up face, soft lighting, looking at camera"
```

### 4.6 Запустить тренировку

```bash
cd /workspace/ai-toolkit
python run.py config/train_misu.yaml
```

**Что ты увидишь:**
```
Step 250/2000 | Loss: 0.0834 | LR: 0.000148
  → Saving checkpoint...
  → Generating sample images...
Step 500/2000 | Loss: 0.0612 | LR: 0.000135
  ...
```

**Время:**
- A6000 48GB: ~30-40 минут
- A100 80GB: ~15-20 минут

### 4.7 Выбрать лучший чекпоинт

После тренировки у тебя будет:
```
/workspace/output/misu_face_lora/
├── misu_face_lora_000000250.safetensors
├── misu_face_lora_000000500.safetensors
├── misu_face_lora_000000750.safetensors
├── ...
├── misu_face_lora_000002000.safetensors   ← последний
└── samples/
    ├── sample_000250_0.png   ← превью на 250 шаге
    ├── sample_000500_0.png
    └── ...
```

**Посмотри sample PNG файлы** — выбери шаг где лицо наиболее похоже и при этом 
изображение не "burnt" (не пересыщенное). Обычно лучший результат между step 1000-1500.

### 4.8 Скачать LoRA

Скачай лучший чекпоинт через RunPod File Manager. Переименуй в `misu.safetensors`.

### 4.9 Повторить для остальных моделей

Для каждой модели:
1. Загрузи фотки в `/workspace/datasets/jane/` (и т.д.)
2. Скопируй конфиг, поменяй:
   - `name: "jane_face_lora"`
   - `trigger_word: "jane"`
   - `folder_path: "/workspace/datasets/jane"`
   - sample prompts с `jane`
3. Запусти `python run.py config/train_jane.yaml`

```bash
# Быстрый способ — копируй и меняй:
cp config/train_misu.yaml config/train_jane.yaml
sed -i 's/misu/jane/g' config/train_jane.yaml
python run.py config/train_jane.yaml
```

> **Совет**: Не останавливай pod между тренировками! Модель уже загружена в VRAM,
> каждая следующая тренировка стартует быстрее.

---

## 🎯 Часть 5: Использование LoRA в боте

### 5.1 Загрузить на RunPod Serverless volume

```bash
# Положи все LoRA файлы в volume:
/runpod-volume/ComfyUI/models/loras/
├── misu.safetensors       (~200-500 MB)
├── jane.safetensors
├── anya.safetensors
├── lera.safetensors
├── mirana.safetensors
└── moondina.safetensors
```

### 5.2 Как использовать в промпте

**Одна девушка:**
```
misu, a young woman sitting on bed, wearing white lingerie, soft natural lighting
```
LoRA strength: 0.7-0.9

**Две девушки (загрузи обе LoRA):**
```
misu and jane, two young women passionately kissing, topless, bedroom setting
```
Каждая LoRA strength: 0.6-0.7 (суммарно не больше ~1.4)

### 5.3 Оптимальный LoRA strength

| Strength | Эффект |
|---|---|
| 0.3–0.5 | Лёгкое сходство. Больше свободы для модели |
| **0.6–0.8** | **✅ Оптимум. Хорошее сходство + гибкость** |
| 0.8–1.0 | Сильное сходство, но меньше гибкости |
| 1.0+ | ⚠️ Может пересыщать. Артефакты |

---

## 💰 Часть 6: Расчёт стоимости

| Модель | GPU | Время | Стоимость |
|---|---|---|---|
| 1 модель на A6000 | 48GB | ~35 мин | ~$0.45 |
| 1 модель на A100 | 80GB | ~18 мин | ~$0.50 |
| **6 моделей на A100** | 80GB | **~2 часа** | **~$3.30** |
| 6 моделей на A6000 | 48GB | ~3.5 часа | ~$2.70 |

> Включи ~10-15 мин на setup → итого $4-5 за все 6 LoRA.

---

## 📋 Чеклист перед стартом

- [ ] 15-25 фоток каждой модели в отдельных папках
- [ ] Фотки: лицо видно, разные ракурсы, разная одежда, разный фон
- [ ] Убраны: без лица, дубли, водяные знаки, low-res
- [ ] .txt caption для КАЖДОЙ фотки (trigger word + описание)
- [ ] Captions проверены вручную
- [ ] RunPod аккаунт с балансом
- [ ] Доступ к flux-2-klein-9b.safetensors
