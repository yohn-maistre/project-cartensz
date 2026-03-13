"""
Bungkus LiteLLM untuk rotasi API key cerdas dan manajemen limit permintaan.
Merutekan semua panggilan LLM melalui litellm untuk mendukung akses terhadap
model seperti Gemini dan Qwen secara fleksibel.
"""
import os
import hashlib
import json
from pathlib import Path
from typing import Optional

import litellm

# matikan log verbose
litellm.suppress_debug_info = True
litellm.drop_params = True

# --- direktori penyimpan cache ---
CACHE_DIR = Path(__file__).parent.parent / "data" / ".llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(model: str, messages: list, **kwargs) -> str:
    """buat identitas cache berdasarkan isi request."""
    payload = json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(key: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, response_text: str):
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps({"content": response_text}), encoding="utf-8")


def get_gemini_keys() -> list[str]:
    """ambil api keys dari env."""
    keys = []
    base = os.getenv("GEMINI_API_KEY", "")
    if base:
        keys.append(base)
    # rotasi variabel lainnya
    for i in range(2, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if k:
            keys.append(k)
    return keys


_current_key_index = 0


def _next_key() -> str:
    """beralih kunci secara rotasi berurutan."""
    global _current_key_index
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("kunci akun gemini tidak ditemukan dalam sistem.")
    key = keys[_current_key_index % len(keys)]
    _current_key_index += 1
    return key


def llm_completion(
    prompt: str,
    system_prompt: str = "",
    model: str = "gemini/gemini-3-flash-preview",
    temperature: float = 0.1,
    use_cache: bool = True,
    **kwargs,
) -> str:
    """
    kirim permintaan melalui litellm memakai rotasi.
    mengirim balik teks dari respons agen llm.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # verifikasi cache
    if use_cache:
        cache_k = _cache_key(model, messages, temperature=temperature)
        cached = _load_cache(cache_k)
        if cached:
            return cached["content"]

    api_key = _next_key()

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content

        # simpan luaran
        if use_cache:
            _save_cache(cache_k, content)

        return content

    except litellm.exceptions.RateLimitError:
        # ganti ke sambungan baru
        api_key = _next_key()
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        if use_cache:
            _save_cache(cache_k, content)
        return content
