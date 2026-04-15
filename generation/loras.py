"""Character LoRA detection — trigger word → LoRA filename + strength."""

import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Character LoRAs — trigger word → LoRA filename + strength
# Add new characters here after training. Trigger word must be lowercase.
# -------------------------------------------------------------------------
CHARACTER_LORAS = {
    "misu": {"lora_name": "misu_z6_step1000.safetensors", "strength": 0.95},
    "anya":  {"lora_name": "anya_lora.safetensors", "strength": 0.9},
    "jane":  {"lora_name": "janelora.safetensors", "strength": 0.9},
    "lera":  {"lora_name": "leralora.safetensors", "strength": 0.9},
    "mirana": {"lora_name": "miranalora.safetensors", "strength": 0.9},
    "moondina": {"lora_name": "moonlora.safetensors", "strength": 0.9},
    "rina":  {"lora_name": "rinalora.safetensors", "strength": 0.9},
}


def detect_character_loras(prompt: str) -> list[dict]:
    """Detect character trigger words in prompt, return list of LoRA configs."""
    prompt_lower = prompt.lower()
    found = []
    for trigger, cfg in CHARACTER_LORAS.items():
        if trigger in prompt_lower:
            found.append({"trigger": trigger, **cfg})
            logger.info("Detected character LoRA: %s → %s", trigger, cfg["lora_name"])
    return found
