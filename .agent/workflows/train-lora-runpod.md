---
description: How to train Face LoRA for Z-Image Turbo (DarkBeastZ6) on RunPod
---

# 🎓 Face LoRA Training — Полный Гайд
## Z-Image Turbo (DarkBeastZ6) | ai-toolkit | RunPod

> Источники: ai-toolkit docs, HuggingFace, CivitAI, Reddit, ZImage.run,
> китайское сообщество (AiMetatron GY, BMX 726, hinablue.me, cnblogs.com),
> modelscope.cn, runcomfy.com.

---

## ⚠️ КРИТИЧЕСКОЕ ОТЛИЧИЕ от Flux Klein

Z-Image Turbo — **дистиллированная** модель. Прямая тренировка без адаптера
**сломает** Turbo-ускорение (модель начнёт требовать 50+ шагов вместо 8).

**Решение**: ai-toolkit имеет специальный **de-distillation training adapter**,
который "де-дистиллирует" модель на время тренировки, позволяя LoRA учиться
без потери Turbo-скорости. После тренировки адаптер убирается.

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
├── 2-3 NSFW (если нужно чтобы LoRA знала тело)
└── 1-2 с видимыми татуировками (если есть!)

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

### 1.4 Про татуировки

⚠️ **Татуировки — самое сложное для LoRA на Z-Image Turbo!**

По данным Reddit и китайского сообщества:
- Z-Image Turbo **рандомизирует** татуировки даже при большом датасете
- SDXL LoRA справляются с татуировками лучше
- Для Z-Image Turbo нужно:
  - **Больше фоток** с чётко видимыми тату (8-15 фоток ТОЛЬКО тату крупным планом)
  - **Детальные captions**: "forearm tattoo of a black rose", "chest tattoo of a dragon"
  - **Rank 128+** для захвата деталей тату
  - Возможно потребуется **отдельная LoRA** только для тату
  - Как альтернатива — использовать ControlNet для позиционирования тату

### 1.5 Captioning (описания) — САМАЯ ВАЖНАЯ ЧАСТЬ

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

### 1.6 Авто-captioning

// turbo
Запусти мой скрипт `auto_caption.py`:
```bash
pip install transformers torch pillow
python auto_caption.py --dir "D:\AI TRAIN\Misu" --trigger "misu"
```

Это создаст `.txt` файлы автоматически с помощью BLIP, но **ОБЯЗАТЕЛЬНО проверь и подправь** каждый caption вручную! Автоматические описания часто неточные.

### 1.7 Финальная структура папки

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
| **Rank (network_dim)** | **128** | Для Z-Image Turbo нужно 128 чтобы хватило "памяти" для лица + детали тела |
| **Alpha (network_alpha)** | **128** или **64** | alpha=rank → нейтральный эффект. alpha=rank/2 → более стабильно |
| **Learning Rate** | **1e-4** | Для Z-Image Turbo 1e-4 оптимально. Выше — нестабильно |
| **LR Scheduler** | **cosine** | Плавно уменьшает LR к концу тренировки |
| **Steps** | **2500–3500** | Для Z-Image Turbo нужно чуть больше шагов чем Klein |
| **Batch Size** | **1–2** | 1 для <= 24GB VRAM, 2 для 48GB+ |
| **Optimizer** | **adamw8bit** | 8-bit AdamW экономит VRAM без потери качества |
| **Resolution** | **1024×1024** | Нативное разрешение Z-Image Turbo |
| **Gradient Checkpointing** | **true** | Экономит VRAM ценой ~20% замедления |
| **Mixed Precision** | **bf16** | fp32 ещё лучше для качества (если влезет) |
| **Quantize base model** | **true** | Для <= 48GB VRAM. На 80GB можно false |
| **Training Adapter** | **v1** | ⚠️ КРИТИЧНО! De-distillation adapter |

### 2.2 Формула расчёта steps

```
Рекомендуемые steps = кол-во фоток × repeats × epochs

Пример:
  20 фоток × 30 repeats × 3 epochs = 1800 steps
  → Для Z-Image Turbo += 50%, итого ~2700 steps

Сохранение каждые 250-500 steps → потом выбираешь лучший чекпоинт
```

### 2.3 Что НЕ тренировать

| Компонент | Тренировать? | Почему |
|---|---|---|
| **UNet/DiT** | ✅ Да | Основная визуальная модель — здесь "живёт" лицо |
| **Text Encoder** | ❌ Нет | Для Z-Image Turbo text encoder НЕ тренируется |

### 2.4 Caption dropout

Установи `caption_dropout_rate: 0.05` — в 5% случаев caption будет пропускаться. 
Это учит LoRA ассоциировать trigger word с лицом даже без описания.

### 2.5 De-distillation Training Adapter

⚠️ **САМОЕ ВАЖНОЕ ОТЛИЧИЕ от тренировки на Flux/Klein/SDXL!**

Z-Image Turbo — дистиллированная модель. Без адаптера:
- LoRA "сломает" CFG=1 / 8-step режим
- Изображения станут размытыми при низком числе шагов
- Потребуется 30-50 шагов вместо 8

**ai-toolkit автоматически** загружает training adapter если выбран Z-Image Turbo.
Но убедись что:
1. `is_z_image_turbo: true` в конфиге (или выбран через UI)
2. Adapter скачается автоматически при первом запуске
3. Есть два варианта: **v1** (стабильный, по умолчанию) и **v2** (экспериментальный)

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
3. ✅ LR не выше 1.5e-4
4. ✅ Не больше 3500 steps для 20 фоток
5. ✅ caption_dropout_rate: 0.05
6. ✅ Cosine LR scheduler

### 3.2 "Turbo Drift" — ТОЛЬКО для Z-Image Turbo

**Признаки:**
- LoRA работает только на 30+ шагах
- При 8 шагах картинка размытая/мусорная
- Пришлось поднять CFG > 1 чтобы что-то получилось

**Причина:** Тренировка без de-distillation adapter!

**Решение:**
- Используй ai-toolkit с `is_z_image_turbo: true`
- Adapter загрузится автоматически
- НИКОГДА не тренируй Z-Image Turbo напрямую без адаптера

### 3.3 Частые ошибки

| Ошибка | Последствие | Решение |
|---|---|---|
| Все фотки на одном фоне | LoRA привязывается к фону | Разные фоны |
| Описал лицо в caption | LoRA не учит лицо как "identity" | Не описывай лицо |
| Rank слишком маленький (8-16) | Не хватает "памяти" для лица | Минимум 64, лучше 128 |
| Слишком много steps | Переобучение | 2500-3500 для 20 фоток |
| Не проверял чекпоинты | Не знаешь какой лучший | save_every: 250 |
| Забыл trigger word в caption | LoRA не привязана к слову | Всегда в начале |
| Фотки low-res (256px) | Модель учит размытые лица | Минимум 512, лучше 1024 |
| **Без training adapter** | **Turbo drift! LoRA ломает 8-step** | **is_z_image_turbo: true** |

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

Нужен Z-Image Turbo base model:

```bash
# Скачать Z-Image Turbo с HuggingFace
cd /workspace/models/
pip install huggingface_hub
huggingface-cli download Comfy-Org/z_image_turbo \
    split_files/diffusion_models/z_image_turbo.safetensors \
    --local-dir /workspace/models/
```

Или если model уже есть на RunPod volume:
```bash
cp /runpod-volume/ComfyUI/models/diffusion_models/z_image_turbo_base*.safetensors /workspace/models/
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
        steps: 3000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        lr_scheduler: "cosine"
        max_denoising_steps: 50
        dtype: bf16
      model:
        name_or_path: "/workspace/models/z_image_turbo.safetensors"
        is_z_image_turbo: true
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

> ⚠️ **Ключевое отличие от Klein**: `is_z_image_turbo: true` — это включает
> de-distillation training adapter автоматически!
> Если поле отсутствует, используй UI ai-toolkit и выбери "Z-Image Turbo".

### 4.6 Запустить тренировку

```bash
cd /workspace/ai-toolkit
python run.py config/train_misu.yaml
```

**Что ты увидишь:**
```
Loading de-distillation training adapter v1...
Step 250/3000 | Loss: 0.0834 | LR: 0.000098
  → Saving checkpoint...
  → Generating sample images...
Step 500/3000 | Loss: 0.0612 | LR: 0.000089
  ...
```

**Время:**
- A6000 48GB: ~40-50 минут
- A100 80GB: ~20-25 минут

### 4.7 Выбрать лучший чекпоинт

После тренировки у тебя будет:
```
/workspace/output/misu_face_lora/
├── misu_face_lora_000000250.safetensors
├── misu_face_lora_000000500.safetensors
├── misu_face_lora_000000750.safetensors
├── ...
├── misu_face_lora_000003000.safetensors   ← последний
└── samples/
    ├── sample_000250_0.png   ← превью на 250 шаге
    ├── sample_000500_0.png
    └── ...
```

**Посмотри sample PNG файлы** — выбери шаг где лицо наиболее похоже и при этом 
изображение не "burnt" (не пересыщенное). Обычно лучший результат между step 1500-2500.

### 4.8 Скачать LoRA

Скачай лучший чекпоинт через RunPod File Manager. Переименуй в `misu_lora.safetensors`.

### 4.9 Загрузить на RunPod Serverless volume

```bash
# На serverless volume:
cp misu_face_lora_BEST.safetensors /runpod-volume/ComfyUI/models/loras/misu_lora.safetensors
```

### 4.10 Повторить для остальных моделей

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
├── misu_lora.safetensors      (~200-500 MB)
├── jane_lora.safetensors
├── anya_lora.safetensors
├── lera_lora.safetensors
├── mirana_lora.safetensors
└── moondina_lora.safetensors
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
| 1 модель на A6000 | 48GB | ~45 мин | ~$0.57 |
| 1 модель на A100 | 80GB | ~22 мин | ~$0.60 |
| **6 моделей на A100** | 80GB | **~2.5 часа** | **~$4.10** |
| 6 моделей на A6000 | 48GB | ~4.5 часа | ~$3.40 |

> Включи ~10-15 мин на setup → итого $4-5 за все 6 LoRA.

---

## 📋 Чеклист перед стартом

- [ ] 15-25 фоток каждой модели в отдельных папках
- [ ] Фотки: лицо видно, разные ракурсы, разная одежда, разный фон
- [ ] Включены фотки с татуировками крупным планом (если есть)
- [ ] Убраны: без лица, дубли, водяные знаки, low-res
- [ ] .txt caption для КАЖДОЙ фотки (trigger word + описание)
- [ ] Captions проверены вручную
- [ ] RunPod аккаунт с балансом
- [ ] Z-Image Turbo base model доступна
