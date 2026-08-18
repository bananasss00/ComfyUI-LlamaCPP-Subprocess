import os
import subprocess
import time
import urllib.request
import urllib.error
import json
import base64
import socket
import re
import wave
import numpy as np
from PIL import Image
import io
import fnmatch
import platform
import zipfile
import threading
import inspect
import atexit
import shlex
import execution
import unicodedata

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import folder_paths
from comfy_api.latest import ComfyExtension
from comfy_api.latest import io as comfy_io  # алиас, чтобы не затенять стандартный модуль `io`
import comfy.model_management

def sanitize_prompt(text: str) -> str:
    if not text:
        return ""
    
    # 1. Юникод-нормализация NFKC
    # Конвертирует полноширинные буквы (ｋｅｙ -> key), математические начертания и лигатуры в стандартный ASCII
    normalized = unicodedata.normalize('NFKC', text)
    
    # 2. Удаление невидимых символов и управляющих кодов юникода
    # \u200b-\u200d: невидимые пробелы нулевой ширины и соединители
    # \ufeff: маркер порядка байтов (BOM)
    # \u200e\u200f: маркеры направления текста LTR/RTL
    clean = re.sub(r'[\u200b-\u200d\ufeff\u200e\u200f]', '', normalized)
    
    # 3. Удаление ASCII-управляющих символов (кроме табуляции и переноса строки)
    clean = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', clean)
    
    # 4. Схлопывание множественных пробелов в один (если они образовались)
    clean = re.sub(r'[ \t]+', ' ', clean)
    
    return clean.strip()

# =======================================================================
# 1. СИСТЕМА ПУТЕЙ И РЕЕСТР ФАЙЛОВ (DROPDOWNS)
# =======================================================================

LLM_FOLDER = "llm_text_processor_models"
PROMPT_FOLDER = "llm_text_processor_prompts"
NO_SYSTEM_PROMPT = "none"
NO_MMPROJ = "none"
NO_MODELS_FOUND = "No GGUF models found"
NO_DRAFT_MODEL = "none (built-in MTP / N-gram)"

def draft_model_options() -> list[str]:
    models = model_options()
    if NO_MODELS_FOUND in models:
        return [NO_DRAFT_MODEL]
    return [NO_DRAFT_MODEL] + models

# Словарь с базовыми пресетами промптов
DEFAULT_PROMPTS = {
    "prompt_refine_expand.txt": "Refine and enhance the following user prompt for creative text-to-image generation. Keep the meaning and keywords, make it more expressive and visually rich. Output ONLY the improved prompt text (no preface, no bullets, no JSON, no <think>, no commentary).",
    "short_story.txt": "Write a short, imaginative story inspired by this image or video.",
    "video_summary.txt": "Summarize the key events and narrative points in this video.",
    "detailed_analysis.txt": "Output ONLY these sections with short labels (no bullets): Subject; People (if any); Environment; Lighting; Camera/Composition; Color/Texture. In each section, write 2–4 sentences of concrete visible details. If something is not visible, write 'not visible'. No preface, no reasoning, no <think>.",
    "cinematic_description.txt": "Write ONE cinematic paragraph (8–12 sentences). Describe the scene like a film still: subject(s) and action; environment and atmosphere; lighting design (practical lights vs ambient, direction, contrast); camera language (shot type, angle, lens feel, depth of field, motion implied); composition and mood. Keep it vivid but factual (no made-up story). No preface, no reasoning, no <think>.",
    "ultra_detailed_description.txt": "Write ONE ultra-detailed paragraph (10–16 sentences, ~180–320 words). Stay grounded in visible details. Include: subject micro-details (materials, textures, patterns, wear, reflections); people details if present (hair, skin tones, makeup, jewelry, fabric types, fit); environment depth (foreground/midground/background, signage/props, surface materials); lighting analysis (key/fill/back light, direction, softness, highlights, shadow shape); camera perspective (angle, lens feel, depth of field) and composition (leading lines, negative space, symmetry/asymmetry, visual hierarchy). No preface, no reasoning, no <think>.",
    "detailed_description.txt": "Write ONE detailed paragraph (6–10 sentences). Describe only what is visible: subject(s) and actions; people details if present (approx age group, gender expression if clear, hair, facial expression, pose, clothing, accessories); environment (location type, background elements, time cues); lighting (source, direction, softness/hardness, color temperature, shadows); camera viewpoint (eye-level/low/high, distance) and composition (framing, focal emphasis). No preface, no reasoning, no <think>.",
    "simple_description.txt": "Analyze the image and write a single concise sentence that describes the main subject and setting. Keep it grounded in visible details only.",
    "tags.txt": "Your task is to generate a clean list of comma-separated tags for a text-to-image AI, based *only* on the visual information in the image. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags."
}

def llm_root() -> Path:
    return Path(folder_paths.models_dir) / "LLM"

def prompt_root() -> Path:
    return llm_root() / "prompts"

def register_folders() -> None:
    llm_dir = llm_root()
    prompts_dir = prompt_root()
    llm_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Автоматическое создание пресетов, если их нет
    for filename, content in DEFAULT_PROMPTS.items():
        filepath = prompts_dir / filename
        if not filepath.exists():
            try:
                filepath.write_text(content, encoding="utf-8")
                print(f"[LlamaCPP] Создан стандартный пресет: {filename}")
            except Exception as e:
                print(f"[LlamaCPP] Предупреждение: не удалось создать пресет {filename}. Ошибка: {e}")
    
    folder_paths.folder_names_and_paths[LLM_FOLDER] = ([str(llm_dir)], {".gguf"})
    folder_paths.folder_names_and_paths[PROMPT_FOLDER] = ([str(prompts_dir)], {".txt"})

def model_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    models =[name for name in files if "mmproj" not in Path(name).name.lower()]
    return models or [NO_MODELS_FOUND]

def mmproj_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    mmproj =[name for name in files if "mmproj" in Path(name).name.lower()]
    return [NO_MMPROJ] + mmproj

def system_prompt_options() -> list[str]:
    # Прямое сканирование директории, минуя кеш folder_paths.
    # folder_paths.get_filename_list() кеширует результат и не инвалидирует
    # кеш при ручном добавлении файлов в папку, поэтому список не обновлялся
    # при перезагрузке страницы ComfyUI.
    prompts_dir = prompt_root()
    if not prompts_dir.exists():
        return [NO_SYSTEM_PROMPT]
    top_level_files = sorted([
        f.name for f in prompts_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    ])
    return [NO_SYSTEM_PROMPT] + top_level_files

def full_model_path(name: str) -> Path:
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    return Path(path) if path else Path("")

register_folders()

# =======================================================================
# 2. АВТО-СКАЧИВАНИЕ LLAMA-SERVER.EXE
# =======================================================================

LLAMA_CPP_RELEASE_TAG = "b10488"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"
# Локальный кеш списка ассетов релиза. Release-assets на GitHub неизменны,
# поэтому кеш валиден навсегда — после первой загрузки мы вообще не обращаемся к github.com.
RELEASE_CACHE_FILE = VENDOR_ROOT / f"_release_cache_{LLAMA_CPP_RELEASE_TAG}.json"

@dataclass(frozen=True)
class PlatformSpec:
    key: str
    cli_executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]

WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    cli_executable="llama-server.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=("llama-server.exe", "ggml-cuda.dll", "cudart64_13.dll"),
)

def _download_file(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with destination.open("wb") as f:
            while chunk := resp.read(1024 * 256):
                f.write(chunk)


def _fetch_release_assets_via_html(tag: str) -> list:
    """
    Возвращает список ассетов релиза [{"name": ..., "browser_download_url": ...}, ...]
    БЕЗ использования GitHub API — парсит обычную HTML-страницу expanded_assets.
    
    Почему так: GitHub API имеет лимит 60 запросов/час для анонимов и легко бьётся
    при частых перезапусках ComfyUI. HTML-страница /releases/expanded_assets/{tag}
    — это обычная веб-страница, она НЕ попадает под API rate-limit. Токен не нужен.
    
    Ассеты на GitHub неизменны, поэтому результат кешируется локально —
    при следующем запуске мы вообще не обращаемся к github.com.
    """
    # 1. Читаем локальный кеш (валиден навсегда — release-assets на GitHub неизменны)
    if RELEASE_CACHE_FILE.exists():
        try:
            with RELEASE_CACHE_FILE.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached:
                return cached
        except Exception:
            pass  # кеш битый — идём в сеть
    
    # 2. Парсим HTML-страницу github.com/.../releases/expanded_assets/{tag}
    #    Эта страница НЕ попадает под API rate-limit (она для браузеров, не для API).
    url = f"https://github.com/ggml-org/llama.cpp/releases/expanded_assets/{tag}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ComfyUI-LLM",
        "Accept": "text/html,application/xhtml+xml",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"\n[LlamaCPP] Release tag '{tag}' не найден на github.com/ggml-org/llama.cpp.\n"
                f"Проверьте LLAMA_CPP_RELEASE_TAG в __init__.py или скачайте llama-server.exe вручную:\n"
                f"  https://github.com/ggml-org/llama.cpp/releases\n"
                f"  Распаковать в: {VENDOR_ROOT / tag / WINDOWS_CUDA_13.key}\n"
            ) from e
        raise
    
    # 3. Извлекаем URL ассетов из HTML
    #    Pattern: href="/ggml-org/llama.cpp/releases/download/{tag}/{asset_name}"
    assets = []
    pattern = re.compile(
        rf'href="(/ggml-org/llama\.cpp/releases/download/{re.escape(tag)}/([^"]+))"',
        re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        relative_url = match.group(1)
        asset_name = match.group(2)
        full_url = "https://github.com" + relative_url
        assets.append({
            "name": asset_name,
            "browser_download_url": full_url,
        })
    
    if not assets:
        raise RuntimeError(
            f"\n[LlamaCPP] Не удалось найти ассеты на странице {url}.\n"
            f"Возможно, релиз {tag} ещё не выложен или изменился формат страницы.\n"
            f"Скачайте llama-server.exe вручную: https://github.com/ggml-org/llama.cpp/releases/tag/{tag}\n"
            f"Распаковать в: {VENDOR_ROOT / tag / WINDOWS_CUDA_13.key}\n"
        )
    
    # 4. Сохраняем в кеш для следующих запусков
    try:
        VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
        with RELEASE_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(assets, f, ensure_ascii=False)
    except Exception:
        pass  # кеш не записался — не критично
    
    return assets


def ensure_llama_server_paths() -> str:
    system = platform.system().lower()
    if system != "windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Auto-download is only supported on Windows x64. Please specify your executable_path manually.")
    
    spec = WINDOWS_CUDA_13
    install_dir = VENDOR_ROOT / LLAMA_CPP_RELEASE_TAG / spec.key
    exe_path = install_dir / spec.cli_executable

    if exe_path.exists():
        return str(exe_path)

    print(f"[LlamaCPP] Auto-downloading llama-server.exe (release {LLAMA_CPP_RELEASE_TAG})... Please wait.")
    install_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список ассетов через HTML-страницу (без API rate-limit)
    assets = _fetch_release_assets_via_html(LLAMA_CPP_RELEASE_TAG)
    
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for pattern in spec.asset_patterns:
            matches = [a for a in assets if fnmatch.fnmatch(a.get("name", "").lower(), pattern.lower())]
            if matches:
                asset = sorted(matches, key=lambda i: i.get("name", ""))[0]
                archive_path = temp_dir / asset["name"]
                _download_file(asset["browser_download_url"], archive_path)
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(install_dir)
                    
    if not exe_path.exists():
        raise RuntimeError(f"Failed to find {spec.cli_executable} after download.")

    # === АВТО-ЧИСТКА СТАРЫХ ВЕРСИЙ ===
    # После успешной загрузки текущей (hardcoded) версии удаляем все остальные
    # подкаталоги в VENDOR_ROOT и старые кеш-файлы _release_cache_*.json,
    # кроме тех, что соответствуют LLAMA_CPP_RELEASE_TAG.
    # Это предотвращает накопление старых релизов на диске.
    try:
        if VENDOR_ROOT.exists():
            import shutil
            removed = []
            current_cache_name = RELEASE_CACHE_FILE.name
            for sub in VENDOR_ROOT.iterdir():
                # Подкаталоги релизов (например b9859/, b10354/)
                if sub.is_dir() and sub.name != LLAMA_CPP_RELEASE_TAG:
                    shutil.rmtree(sub, ignore_errors=True)
                    removed.append(sub.name)
                # Старые кеш-файлы _release_cache_*.json (кроме текущего)
                elif sub.is_file() and sub.name.startswith("_release_cache_") and sub.name.endswith(".json") and sub.name != current_cache_name:
                    try:
                        sub.unlink()
                        removed.append(sub.name)
                    except Exception:
                        pass
            if removed:
                print(f"[LlamaCPP] Removed {len(removed)} old llama.cpp artifact(s): {', '.join(removed)}")
    except Exception as cleanup_err:
        print(f"[LlamaCPP Warning] Failed to clean old llama.cpp versions: {cleanup_err}")

    return str(exe_path)


def resolve_executable_path(executable_path: str) -> str:
    """
    Разрешает путь к llama-server.exe.
    Если executable_path указывает на существующий файл — используем его.
    Иначе (включая 'auto', пустую строку, опечатку, удалённый файл) — встроенная llama.
    """
    raw = (executable_path or "").strip()
    if raw and os.path.isfile(raw):
        return raw
    return ensure_llama_server_paths()

# =======================================================================
# 3. ОСНОВНАЯ ЛОГИКА И СЕРВЕРЫ
# =======================================================================

ACTIVE_SERVERS = {}
CHAT_SESSIONS = {}  # Глобальный реестр для хранения истории чат-сессий
ORIGINAL_EXTRA_RESERVED_VRAM = None
MMPROJ_EMBEDDING_MISMATCH_RE = re.compile(
    r"mismatch between text model \(n_embd = (?P<model>\d+)\) and mmproj \(n_embd = (?P<mmproj>\d+)\)", flags=re.IGNORECASE
)

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
ANY = AnyType("*")

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def normalize_llama_seed(seed: int) -> int:
    seed = int(seed)
    if seed <= 0: return 42 # Fallback to 42 for random/invalid to avoid overflow crashes
    return seed % (2**32)

def tensors_to_base64_list(image_tensor, max_frames=8):
    total_frames = image_tensor.shape[0]
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        
    b64_list =[]
    for i in indices:
        img_np = 255. * image_tensor[i].cpu().numpy()
        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        b64_list.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return b64_list

def audio_to_base64_wav(audio_input) -> str:
    if not isinstance(audio_input, dict) or "waveform" not in audio_input:
        raise ValueError("Неверный формат AUDIO в ComfyUI.")
        
    waveform = audio_input["waveform"]
    sample_rate = audio_input.get("sample_rate", 16000)
    
    # ComfyUI AUDIO тензор имеет форму [batch, channels, samples]
    if waveform.dim() == 3:
        waveform = waveform[0] # Берем первый батч
        
    audio_np = waveform.cpu().numpy()
    
    # Приводим к размерности [samples, channels]
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    elif audio_np.ndim == 1:
        audio_np = audio_np[:, np.newaxis]
        
    # Нормализуем float (-1.0 ... 1.0) в 16-битный PCM (signed int16)
    audio_int16 = np.clip(audio_np * 32767.0, -32768.0, 32767.0).astype(np.int16)
    num_channels = audio_int16.shape[1]
    
    # Записываем WAV в оперативную память
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
        
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def update_global_vram_reservation():
    global ACTIVE_SERVERS, ORIGINAL_EXTRA_RESERVED_VRAM
    if ORIGINAL_EXTRA_RESERVED_VRAM is None:
        ORIGINAL_EXTRA_RESERVED_VRAM = getattr(comfy.model_management, "EXTRA_RESERVED_VRAM", 0)
    
    total_reserve_bytes = 0
    for s_id, s_info in ACTIVE_SERVERS.items():
        s_vram = s_info.get("vram_bytes", 0)
        s_extra_reserve_vram_gb = s_info.get("extra_reserve_vram", 0.6)
        s_extra_bytes = int(s_extra_reserve_vram_gb * (1024 ** 3))
        total_reserve_bytes += (s_vram + s_extra_bytes)
        
    if total_reserve_bytes > 0:
        comfy.model_management.EXTRA_RESERVED_VRAM = total_reserve_bytes
        print(f"[LlamaCPP] Суммарно зарезервировано VRAM для всех серверов: {total_reserve_bytes / (1024**3):.2f} GB")
    else:
        if ORIGINAL_EXTRA_RESERVED_VRAM is not None:
            try:
                comfy.model_management.EXTRA_RESERVED_VRAM = ORIGINAL_EXTRA_RESERVED_VRAM
                print(f"[LlamaCPP] Все серверы остановлены. Восстановлено исходное значение EXTRA_RESERVED_VRAM: {ORIGINAL_EXTRA_RESERVED_VRAM / (1024**3):.2f} GB")
            except Exception:
                pass
            ORIGINAL_EXTRA_RESERVED_VRAM = None

def kill_server(server_id: str):
    global ACTIVE_SERVERS
    if server_id in ACTIVE_SERVERS:
        s_info = ACTIVE_SERVERS[server_id]
        if s_info.get("process"):
            try:
                s_info["process"].kill()
                s_info["process"].wait(timeout=5)
            except Exception: pass
        if s_info.get("log_file"):
            try: s_info["log_file"].close()
            except Exception: pass
        del ACTIVE_SERVERS[server_id]
        print(f"[LlamaCPP] Сервер '{server_id}' выгружен, процесс завершен.")
        update_global_vram_reservation()

def kill_all_servers():
    global ACTIVE_SERVERS
    for server_id in list(ACTIVE_SERVERS.keys()):
        kill_server(server_id)

# Убиваем все процессы при полном закрытии ComfyUI
atexit.register(kill_all_servers)

if not hasattr(comfy.model_management, "_original_unload_all_models_llamacpp"):
    comfy.model_management._original_unload_all_models_llamacpp = comfy.model_management.unload_all_models

    def hooked_unload_all_models_llamacpp(*args, **kwargs):
        try:
            stack = inspect.stack()
            # Проверяем, вызвано ли это прямо во время выполнения графа (например, нодой Painter VRAM)
            is_node_execution = any("execution.py" in frame.filename for frame in stack)
        except Exception:
            is_node_execution = False

        if not is_node_execution:
            kill_all_servers()
            
        return comfy.model_management._original_unload_all_models_llamacpp(*args, **kwargs)
    
    comfy.model_management.unload_all_models = hooked_unload_all_models_llamacpp

# Перехват запуска графа: Авто-очистка при переключении на воркфлоу без Llama
if not hasattr(execution.PromptExecutor, "_original_execute_llamacpp"):
    execution.PromptExecutor._original_execute_llamacpp = execution.PromptExecutor.execute

    def hooked_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        # Ищем наши ноды в списке узлов, которые сейчас будут выполняться
        active_server_ids_in_prompt = set()
        if isinstance(prompt, dict):
            for node_id, node_info in prompt.items():
                if isinstance(node_info, dict) and node_info.get("class_type") == "LlamaCPP_Subprocess":
                    inputs = node_info.get("inputs", {})
                    s_id = inputs.get("server_id", "default")
                    active_server_ids_in_prompt.add(s_id)
        
        # Если мы нажали "Queue Prompt", а каких-то серверов в этом воркфлоу НЕТ —
        # значит мы переключились на другой процесс или убрали эти ноды. Смело выгружаем лишние модели!
        global ACTIVE_SERVERS
        for s_id in list(ACTIVE_SERVERS.keys()):
            if s_id not in active_server_ids_in_prompt:
                print(f"[LlamaCPP] Сервер '{s_id}' отсутствует в текущем воркфлоу. Выгрузка...")
                kill_server(s_id)

        return self._original_execute_llamacpp(prompt, prompt_id, extra_data, execute_outputs)

    execution.PromptExecutor.execute = hooked_execute

# =======================================================================
# 4. УЗЛЫ ИНТЕРФЕЙСА
# =======================================================================

class LlamaCPPAdvancedSamplersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dynatemp_range": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Динамическая температура (0 = выкл). Позволяет модели быть креативной, снижая предсказуемость, но не сходя с ума на очевидных словах."
                }),
                "dynatemp_exponent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Степень (форма кривой) для динамической температуры. Обычно оставляют 1.0 (линейная)."
                }),
                "xtc_probability": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Вероятность XTC (Exclude Top Choices). С какой вероятностью алгоритм исключит самые ожидаемые (популярные) слова, заставляя модель строить оригинальные фразы."
                }),
                "xtc_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Порог XTC. Исключаются только те слова, базовая вероятность которых выше этого значения (например, выше 10%)."
                }),
                "smoothing_factor": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Сглаживание логитов (Logit Smoothing). Делает распределение выбора слов менее резким. 0 = выключено."
                }),
                "smoothing_curve": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Форма кривой для алгоритма сглаживания вероятностей."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Динамический порог (Min-P). Значение 0.05 отбросит все слова, вероятность которых меньше 5% от вероятности самого подходящего слова. Отличная замена Top-K и Top-P."
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Штраф за присутствие. Заставляет модель избегать повторения уже затронутых тем и слов. Повышает разнообразие текста."
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Штраф за частоту. Чем чаще слово уже встречалось в тексте, тем сильнее будет штраф. Полезно для избавления от слов-паразитов."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Штраф за повторения (Repetition Penalty). Значение 1.0 отключает штраф. Значения выше (например 1.1) заставляют модель реже зацикливаться на одних и тех же фразах."
                }),
                "mirostat": ("INT", {
                    "default": 0, "min": 0, "max": 2, "step": 1,
                    "tooltip": "Mirostat (0 = выкл, 1 = v1, 2 = v2). Умный алгоритм, который автоматически поддерживает качество текста на одном уровне. При использовании лучше отключать Top-K/Top-P."
                }),
                "mirostat_tau": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Целевая энтропия Mirostat (Tau). Чем выше значение, тем более неожиданным (креативным) будет текст. Обычно используют 5.0."
                }),
                "mirostat_eta": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Скорость обучения Mirostat (Eta). Как быстро алгоритм подстраивается под изменения текста. Обычно используют 0.1."
                }),
                "tfs_z": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Tail Free Sampling (TFS). 1.0 = выключено. Плавно отсекает 'хвост' из самых маловероятных слов, менее агрессивен, чем Top-P."
                }),
                "typical_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Typical Sampling. 1.0 = выключено. Алгоритм пытается выбирать слова, вероятность которых близка к ожидаемой естественной случайности контекста."
                }),
            },
            "optional": {
                "banned_tokens": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Список запрещенных слов или фраз (каждая с новой строки). Нода переведет их в logit_bias, и модель никогда их не напишет."
                }),
            }
        }

    RETURN_TYPES = ("LLAMA_SAMPLERS",)
    RETURN_NAMES = ("samplers",)
    FUNCTION = "get_samplers"
    CATEGORY = "LlamaCPP/Inference"

    def get_samplers(self, dynatemp_range, dynatemp_exponent, xtc_probability, xtc_threshold, smoothing_factor, smoothing_curve, min_p, presence_penalty, frequency_penalty, repeat_penalty, mirostat, mirostat_tau, mirostat_eta, tfs_z, typical_p, banned_tokens=""):
        return ({
            "dynatemp_range": dynatemp_range,
            "dynatemp_exponent": dynatemp_exponent,
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
            "smoothing_factor": smoothing_factor,
            "smoothing_curve": smoothing_curve,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repeat_penalty": repeat_penalty,
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "tfs_z": tfs_z,
            "typical_p": typical_p,
            "banned_tokens": banned_tokens
        },)

class LlamaCPPSpeculativeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "draft_model": (draft_model_options(), {
                    "tooltip": "Файл GGUF вспомогательной драфт-модели. Выберите 'none (built-in MTP / N-gram)', если используете встроенный MTP или n-gram спекуляцию."
                }),
                "spec_type": (["draft-mtp", "draft-simple", "draft-eagle3", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache", "none"], {
                    "default": "draft-mtp",
                    "tooltip": "Тип спекулятивного декодирования. Например, 'draft-mtp' для MTP-моделей (Gemma 4 MTP), 'draft-simple' для стандартных легких моделей-ассистентов."
                }),
                "spec_draft_n_max": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Максимальное количество токенов, генерируемых вспомогательной моделью за одну итерацию."
                }),
                "spec_draft_n_min": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Минимальное количество токенов для драфт-модели."
                }),
                "spec_draft_p_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Минимальный порог вероятности спекулятивного токена (0.0 — отключено)."
                }),
                "gpu_layers_draft": ("INT", {
                    "default": 99, "min": -1, "max": 999,
                    "tooltip": "Количество слоев вспомогательной модели, переносимых в VRAM GPU (-ngld). 99 означает полный перенос."
                })
            }
        }

    RETURN_TYPES = ("LLAMA_SPEC_SETTINGS",)
    RETURN_NAMES = ("spec_settings",)
    FUNCTION = "get_settings"
    CATEGORY = "LlamaCPP/Inference"

    def get_settings(self, draft_model, spec_type, spec_draft_n_max, spec_draft_n_min, spec_draft_p_min, gpu_layers_draft):
        return ({
            "draft_model": draft_model,
            "spec_type": spec_type,
            "spec_draft_n_max": spec_draft_n_max,
            "spec_draft_n_min": spec_draft_n_min,
            "spec_draft_p_min": spec_draft_p_min,
            "gpu_layers_draft": gpu_layers_draft
        },)
    
class LlamaCPPChatHistoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {
                    "default": "session_1",
                    "tooltip": "Уникальный идентификатор сессии чата. Измените имя, чтобы начать независимый параллельный диалог."
                }),
                "reset_session": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Очистить историю этой сессии чата и начать заново с системным промптом."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "Системная роль и инструкции для модели в рамках этого чата."
                }),
                "context_mode": (["keep_all", "prune_middle", "auto_summarize"], {
                    "default": "keep_all",
                    "tooltip": "Режим контроля контекста: keep_all (помнить всё), prune_middle (удалять середину), auto_summarize (умное сжатие в архив)."
                }),
                "keep_first_n": ("INT", {
                    "default": 2, "min": 1, "max": 10,
                    "tooltip": "Сколько самых первых сообщений (включая картинку) НИКОГДА не удалять."
                }),
                "keep_recent_n": ("INT", {
                    "default": 4, "min": 1, "max": 20,
                    "tooltip": "Сколько самых свежих сообщений перед новым запросом оставлять нетронутыми."
                }),
                "summarize_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Summarize the following conversation in detail. Preserve key facts, visual descriptions, character rules, and exact quotes if important.",
                    "tooltip": "Промпт для режима auto_summarize (на английском). Укажите, что именно модели нужно сохранить при архивации истории."
                }),
                "summary_max_tokens": ("INT", {
                    "default": 500, "min": 50, "max": 4096,
                    "tooltip": "Максимальный размер архива (саммари) в токенах."
                })
            }
        }

    RETURN_TYPES = ("LLAMA_CHAT_HISTORY", "STRING")
    RETURN_NAMES = ("chat_history", "formatted_history")
    FUNCTION = "get_history"
    CATEGORY = "LlamaCPP/Chat"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Принудительно заставляем ComfyUI обновлять эту ноду каждый раз,
        # чтобы она подтягивала актуальные сообщения из памяти, а не из кэша.
        return float("NaN")

    def get_history(self, session_id, reset_session, system_prompt, context_mode="keep_all", keep_first_n=2, keep_recent_n=4, summarize_prompt="", summary_max_tokens=500):
        global CHAT_SESSIONS
        s_id = session_id.strip() or "default"
        
        if reset_session or s_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[s_id] = []
            if system_prompt.strip():
                CHAT_SESSIONS[s_id].append({
                    "role": "system", 
                    "content": system_prompt.strip()
                })
            print(f"[LlamaCPP] Сессия чата '{s_id}' инициализирована/сброшена.")

        messages = CHAT_SESSIONS[s_id]
        
        formatted_lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                text_content = " ".join(text_parts).strip()
            else:
                text_content = str(content).strip()
                
            if role == "SYSTEM" and len(text_content) > 150:
                text_content = text_content[:150] + "..."
                
            formatted_lines.append(f"[{role}]: {text_content}")
            
        formatted_text = "\n\n".join(formatted_lines)
        
        chat_config = {
            "context_mode": context_mode, "keep_first_n": keep_first_n, 
            "keep_recent_n": keep_recent_n, "summarize_prompt": summarize_prompt, 
            "summary_max_tokens": summary_max_tokens
        }
        
        return ({"session_id": s_id, "messages": messages, "config": chat_config}, formatted_text)


class LlamaCPPFormatHistoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chat_history": ("LLAMA_CHAT_HISTORY", {
                    "tooltip": "Контейнер истории диалога для форматирования в текст."
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_history",)
    FUNCTION = "format_history"
    CATEGORY = "LlamaCPP/Chat"

    def format_history(self, chat_history):
        if not chat_history or "messages" not in chat_history:
            return ("",)
        
        messages = chat_history["messages"]
        formatted_lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                text_content = " ".join(text_parts).strip()
            else:
                text_content = str(content).strip()
                
            formatted_lines.append(f"[{role}]: {text_content}")
            
        return ("\n\n".join(formatted_lines),)


class LlamaCPPSubprocessNode(comfy_io.ComfyNode):
    """
    Llama.cpp subprocess node — migrated to comfy_api.latest.
    Dynamic media inputs (Autogrow): images / videos / audios.
    """

    @classmethod
    def define_schema(cls):
        # --- Autogrow-шаблоны для медиа-входов ---
        image_pin = comfy_io.Image.Input(
            "img",
            optional=True,
            tooltip="Одна картинка ИЛИ батч картинок (IMAGE [N,H,W,C]). Можно подключать сколько угодно.",
        )
        image_template = comfy_io.Autogrow.TemplatePrefix(image_pin, prefix="img", min=0, max=50)

        video_pin = comfy_io.Image.Input(
            "vid",
            optional=True,
            tooltip="Видео как батч последовательных кадров (IMAGE [N,H,W,C]). Можно подключать сколько угодно.",
        )
        video_template = comfy_io.Autogrow.TemplatePrefix(video_pin, prefix="vid", min=0, max=50)

        audio_pin = comfy_io.Audio.Input(
            "aud",
            optional=True,
            tooltip="ComfyUI AUDIO dict (waveform + sample_rate). Можно подключать сколько угодно.",
        )
        audio_template = comfy_io.Autogrow.TemplatePrefix(audio_pin, prefix="aud", min=0, max=50)

        return comfy_io.Schema(
            node_id="LlamaCPP_Subprocess",
            display_name="LlamaCPP Server Model",
            description="Llama.cpp subprocess node with dynamic image / video / audio inputs.",
            category="LlamaCPP/Inference",
            inputs=[
                comfy_io.Combo.Input("model", options=model_options(),
                               tooltip="Файл GGUF модели для текста. Поместите файлы в ComfyUI/models/LLM."),
                comfy_io.Combo.Input("mmproj", options=mmproj_options(), default=NO_MMPROJ,
                               tooltip="Файл проектора зрения (Vision GGUF). Нужен ТОЛЬКО если передаются медиа."),
                comfy_io.String.Input("prompt", multiline=True, default="Опиши это видео или изображения подробно.",
                                tooltip="Текст вашего запроса (промпт)."),
                comfy_io.Int.Input("max_tokens", default=2048, min=1, max=32768, step=1,
                             tooltip="Максимум токенов в ответе."),
                comfy_io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05,
                               tooltip="Температура генерации."),
                comfy_io.Float.Input("top_p", default=0.95, min=0.1, max=1.0, step=0.05,
                               tooltip="Top-P (Nucleus)."),
                comfy_io.Int.Input("top_k", default=40, min=1, max=100, step=1,
                             tooltip="Top-K. Жесткое ограничение выборки."),
                comfy_io.Int.Input("ctx_size", default=16384, min=512, max=128000, step=256,
                             tooltip="Размер контекста в токенах."),
                comfy_io.Boolean.Input("flash_attention", default=True,
                                 tooltip="Flash Attention (-fa). Крайне рекомендуется."),
                comfy_io.Combo.Input("context_quantization",
                               options=["none", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                               default="none",
                               tooltip="Квантование контекста (сжатие KV-cache)."),
                comfy_io.Combo.Input("memory_mode",
                               options=["auto", "gpu_layers", "cpu_moe_layers", "gpu_and_cpu_moe_layers"],
                               default="auto",
                               tooltip="Режим распределения слоев модели."),
                comfy_io.Int.Input("gpu_layers", default=99, min=-1, max=999, step=1,
                             tooltip="Сколько слоев в GPU. 999 = все."),
                comfy_io.Int.Input("n_cpu_moe_layers", default=1, min=1, max=999, step=1,
                             tooltip="MoE-слои на CPU."),
                comfy_io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, step=1,
                             tooltip="Зерно случайности. 0 = случайно."),
                comfy_io.Combo.Input("reasoning", options=["auto", "on", "off"], default="auto",
                               tooltip="Режим размышления (Reasoning)."),
                comfy_io.Boolean.Input("keep_model_loaded", default=True,
                                 tooltip="Оставлять сервер в памяти после ответа."),
                comfy_io.Int.Input("batch_size", default=512, min=64, max=8192, step=64,
                             tooltip="Скорость чтения промпта (--batch-size)."),
                comfy_io.Int.Input("parallel_requests", default=1, min=1, max=8, step=1,
                             tooltip="Количество слотов запросов (-np)."),
                comfy_io.Boolean.Input("no_mmap", default=False, tooltip="Отключить MMap."),
                comfy_io.Boolean.Input("no_warmup", default=False, tooltip="Отключить разогрев."),
                comfy_io.Boolean.Input("mlock", default=False, tooltip="Блокировка в ОЗУ (--mlock)."),
                comfy_io.Int.Input("fit_target_mib", default=0, min=0, max=24000, step=256,
                             tooltip="Гарантированный запас VRAM (-fitt) в МБ."),
                # --- optional ---
                comfy_io.String.Input("override_tensor", default="",
                                tooltip="Хак для видеопамяти (-ot). Например: '.*ffn_down.*=CPU'."),
                comfy_io.Custom("LLAMA_SAMPLERS").Input("extra_samplers", optional=True,
                                                  tooltip="Подключите LlamaCPP Advanced Samplers."),
                comfy_io.Custom("LLAMA_CHAT_HISTORY").Input("chat_history", optional=True,
                                                      tooltip="Подключите историю чата."),
                comfy_io.Custom("LLAMA_SPEC_SETTINGS").Input("spec_settings", optional=True,
                                                      tooltip="Подключите LlamaCPP Speculative Settings."),
                comfy_io.Combo.Input("system_prompt_preset", options=system_prompt_options(), default=NO_SYSTEM_PROMPT,
                               tooltip="Шаблон системного промпта."),
                comfy_io.String.Input("system_prompt_text", multiline=True, default="",
                                tooltip="Дополнительный ручной системный промпт."),
                # --- общий параметр для ВСЕХ видео-пинов ---
                comfy_io.Int.Input("max_video_frames", default=8, min=1, max=128, step=1,
                             tooltip="Сколько кадров равномерно выбрать из каждого видео (через np.linspace, общий для всех video-пинов)."),
                # --- autogrow медиа-входы ---
                comfy_io.Autogrow.Input("images", template=image_template),
                comfy_io.Autogrow.Input("videos", template=video_template),
                comfy_io.Autogrow.Input("audios", template=audio_template),
                # --- misc ---
                comfy_io.String.Input("executable_path", default="auto",
                                tooltip="Путь до llama-server.exe. 'auto' ИЛИ пусто ИЛИ невалидный путь → используется встроенная llama (авто-скачивание). Иначе — указанный вами exe."),
                comfy_io.String.Input("extra_cli_args", default="",
                                tooltip="Дополнительные аргументы командной строки llama.cpp."),
                comfy_io.Float.Input("extra_reserve_vram", default=0.6, min=0.0, max=32.0, step=0.1,
                               tooltip="Дополнительный 'виртуальный' резерв VRAM в ГБ."),
                comfy_io.String.Input("server_id", default="default",
                                tooltip="Уникальный идентификатор сервера."),
                comfy_io.Int.Input("reasoning_budget", default=0, min=0, max=32768, step=128,
                             tooltip="Динамический лимит токенов размышлений (0 = без лимита)."),
                comfy_io.String.Input("reasoning_budget_message", default="Conclusion:",
                                tooltip="Текст, завершающий мысли при превышении лимита."),
            ],
            outputs=[
                comfy_io.String.Output(display_name="text"),
                comfy_io.String.Output(display_name="thoughts"),
                comfy_io.String.Output(display_name="perf"),
                comfy_io.String.Output(display_name="usage_stats"),
                comfy_io.Custom("LLAMA_CHAT_HISTORY").Output(display_name="chat_history"),
            ],
        )

    @classmethod
    def execute(cls,
                # required
                model, mmproj, prompt, max_tokens, temperature, top_p, top_k, ctx_size,
                flash_attention, context_quantization, memory_mode, gpu_layers,
                n_cpu_moe_layers, seed, reasoning, keep_model_loaded,
                batch_size=512, parallel_requests=1, no_mmap=False, no_warmup=False,
                mlock=False, fit_target_mib=0,
                # optional server / sampling / history
                override_tensor="",
                extra_samplers=None, chat_history=None, spec_settings=None,
                system_prompt_preset=NO_SYSTEM_PROMPT, system_prompt_text="",
                # autogrow media + общий параметр видео
                max_video_frames=8,
                images=None, videos=None, audios=None,
                # misc
                executable_path="auto", extra_cli_args="",
                extra_reserve_vram=0.6, server_id="default",
                reasoning_budget=0,
                reasoning_budget_message="Conclusion:") -> comfy_io.NodeOutput:
        
        global ACTIVE_SERVERS, ORIGINAL_EXTRA_RESERVED_VRAM

        prompt = sanitize_prompt(prompt)
        if system_prompt_text:
            system_prompt_text = sanitize_prompt(system_prompt_text)
            
        if model == NO_MODELS_FOUND:
            raise ValueError("No models found. Please put .gguf files in ComfyUI/models/LLM")

        # Resolve paths
        m_path = str(full_model_path(model))
        mm_path = str(full_model_path(mmproj)) if mmproj != NO_MMPROJ else ""
        
        # === Разрешение пути к exe: auto / пусто / невалидный путь → встроенная llama ===
        exe_path = resolve_executable_path(executable_path)

        # Speculative decoding settings
        draft_model_path = ""
        spec_type = "none"
        spec_draft_n_max = 4
        spec_draft_n_min = 0
        spec_draft_p_min = 0.0
        gpu_layers_draft = -1
        
        if spec_settings:
            spec_type = spec_settings.get("spec_type", "none")
            spec_draft_n_max = spec_settings.get("spec_draft_n_max", 4)
            spec_draft_n_min = spec_settings.get("spec_draft_n_min", 0)
            spec_draft_p_min = spec_settings.get("spec_draft_p_min", 0.0)
            gpu_layers_draft = spec_settings.get("gpu_layers_draft", -1)
            
            draft_model_name = spec_settings.get("draft_model", "")
            if draft_model_name and draft_model_name not in {NO_MODELS_FOUND, NO_DRAFT_MODEL}:
                draft_model_path = str(full_model_path(draft_model_name))

        # Create config hash to detect if we need to restart server
        current_config = {
            "exe": exe_path, "model": m_path, "mmproj": mm_path, "ctx": ctx_size, "ctx_q": context_quantization,
            "mem": memory_mode, "gpu": gpu_layers, "moe": n_cpu_moe_layers,
            "args": extra_cli_args, "reasoning": reasoning, "flash_attn": flash_attention,
            "batch": batch_size, "np": parallel_requests, "no_mmap": no_mmap, "no_warmup": no_warmup, 
            "mlock": mlock, "fitt": fit_target_mib, "ot": override_tensor,
            "spec_type": spec_type, "draft_model": draft_model_path, 
            "spec_draft_n_max": spec_draft_n_max, "spec_draft_n_min": spec_draft_n_min,
            "spec_draft_p_min": spec_draft_p_min, "gpu_layers_draft": gpu_layers_draft
        }

        # ЗАЩИТА ОТ КРАШЕЙ (если процесс был убит через диспетчер задач)
        if server_id in ACTIVE_SERVERS:
            proc = ACTIVE_SERVERS[server_id].get("process")
            if proc and proc.poll() is not None:
                print(f"\n[LlamaCPP] Процесс сервера '{server_id}' упал/убит извне (Код {proc.returncode}). Авто-перезапуск...")
                kill_server(server_id)

        if server_id in ACTIVE_SERVERS and ACTIVE_SERVERS[server_id].get("config") != current_config:
            print(f"\n[LlamaCPP] Изменение настроек модели для сервера '{server_id}'. Выгрузка старой...")
            kill_server(server_id)

        # START SERVER IF NEEDED
        if server_id not in ACTIVE_SERVERS:
            port = get_free_port()
            cmd = [exe_path, "-m", m_path, "-c", str(ctx_size), "--port", str(port), "-lv", "4"]
            
            if flash_attention:
                cmd.extend(["-fa", "on"])
                
            cmd.extend(["--batch-size", str(batch_size), "--ubatch-size", str(batch_size)])
            cmd.extend(["-np", str(parallel_requests)])
            
            if no_mmap: cmd.extend(["--no-mmap"])
            if no_warmup: cmd.extend(["--no-warmup"])
            if mlock: cmd.extend(["--mlock"])
            if fit_target_mib > 0: cmd.extend(["-fitt", str(fit_target_mib)])
            if override_tensor and override_tensor.strip(): cmd.extend(["-ot", override_tensor.strip()])
                
            if memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
                cmd.extend(["-ngl", str(gpu_layers)])
            if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
                cmd.extend(["--n-cpu-moe", str(n_cpu_moe_layers)])
            
            if context_quantization and context_quantization != "none":
                cmd.extend(["--cache-type-k", context_quantization, "--cache-type-v", context_quantization])
            
            if mm_path: cmd.extend(["--mmproj", mm_path])
            if spec_type and spec_type != "none":
                cmd.extend(["--spec-type", spec_type])
                
            if draft_model_path:
                cmd.extend(["--model-draft", draft_model_path])
                
            if spec_draft_n_max > 0:
                cmd.extend(["--spec-draft-n-max", str(spec_draft_n_max)])
                
            if spec_draft_n_min > 0:
                cmd.extend(["--spec-draft-n-min", str(spec_draft_n_min)])
                
            if spec_draft_p_min > 0.0:
                cmd.extend(["--spec-draft-p-min", f"{spec_draft_p_min:.3f}"])
                
            if gpu_layers_draft >= 0:
                cmd.extend(["-ngld", str(gpu_layers_draft)])
            if reasoning != "auto": cmd.extend(["--reasoning", reasoning])
            if extra_cli_args and extra_cli_args.strip():
                # Заменяем обратные слэши на прямые до shlex.split, чтобы Python не съедал их на Windows
                clean_extra_args = extra_cli_args.replace('\\', '/')
                parsed_extra_args = [arg.strip('"\'') for arg in shlex.split(clean_extra_args)]
                cmd.extend(parsed_extra_args)

            print(f"\n[LlamaCPP] Запуск сервера '{server_id}': {' '.join(cmd)}")
            log_file_path = os.path.join(os.getcwd(), f"llama_server_{server_id}_debug.log")
            log_file = open(log_file_path, "w", encoding="utf-8")
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="ignore")
            
            # Переменная для перехвата VRAM из потока (используем list для изменения внутри функции)
            startup_vram_mb = [0.0]
            vram_pattern = re.compile(r'(?:VRAM|(?:CUDA|Vulkan|Metal|SYCL)\d+.*?buffer size)[^:\n]*[:=]\s*([\d.]+)\s*MiB', re.IGNORECASE)
            
            def pipe_reader():
                for line in process.stdout:
                    try:
                        log_file.write(line)
                        match = vram_pattern.search(line)
                        if match:
                            startup_vram_mb[0] += float(match.group(1))
                    except Exception:
                        break
                        
            threading.Thread(target=pipe_reader, daemon=True).start()
            
            server_ready = False
            for _ in range(60):
                # Если процесс завершился (например, упал по OOM), прерываем ожидание немедленно
                if process.poll() is not None:
                    break
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
                    with urllib.request.urlopen(req, timeout=1) as response:
                        if response.status == 200:
                            server_ready = True
                            break
                except Exception:
                    time.sleep(1)
            
            if not server_ready:
                process.kill()
                log_file.close()
                with open(log_file_path, "r", encoding="utf-8") as f:
                    err_text = f.read()
                
                # Check for smart error parsing (mmproj mismatch)
                mm_match = MMPROJ_EMBEDDING_MISMATCH_RE.search(err_text)
                if mm_match:
                    raise RuntimeError(
                        f"ВНИМАНИЕ: Выбранный mmproj не подходит к этой текстовой модели! "
                        f"(модель n_embd={mm_match.group('model')}, mmproj n_embd={mm_match.group('mmproj')}). "
                        "Выберите mmproj, соответствующий архитектуре модели."
                    )
                raise Exception(f"[LlamaCPP ERROR] Сервер '{server_id}' не запустился. Лог: {err_text[-1000:]}")
            
            ACTIVE_SERVERS[server_id] = {
                "process": process, 
                "config": current_config, 
                "port": port, 
                "log_file": log_file,
                "vram_bytes": 0,
                "extra_reserve_vram": extra_reserve_vram
            }
            print(f"[LlamaCPP] Сервер '{server_id}' готов на порту {port}.")

            # Даем потоку долю секунды на обработку последних строк перед тем как забрать результат
            time.sleep(0.2)
            exact_vram_bytes = int(startup_vram_mb[0] * 1024 * 1024)

            if exact_vram_bytes > 0:
                ACTIVE_SERVERS[server_id]["vram_bytes"] = exact_vram_bytes
                exact_vram_gb = exact_vram_bytes / (1024 ** 3)
                print(f"[LlamaCPP] Извлечено потребление VRAM для сервера '{server_id}': {exact_vram_gb:.2f} GB")
                update_global_vram_reservation()
            else:
                print(f"[LlamaCPP] Не удалось извлечь потребление VRAM из потока для сервера '{server_id}'. Используются стандартные лимиты памяти.")
                update_global_vram_reservation()

        # ПОДГОТОВКА СТРУКТУРЫ MESSAGES И СЖАТИЕ КОНТЕКСТА
        chat_config = {}
        if chat_history is not None:
            messages = list(chat_history.get("messages", []))
            chat_config = chat_history.get("config", {})
            c_mode = chat_config.get("context_mode", "keep_all")
            k_first = chat_config.get("keep_first_n", 2)
            k_recent = chat_config.get("keep_recent_n", 4)
            
            if c_mode != "keep_all" and len(messages) > k_first + k_recent:
                if c_mode == "prune_middle":
                    messages = messages[:k_first] + messages[-k_recent:]
                    print(f"[LlamaCPP] Контекст усечен (prune_middle). Сохранено сообщений: {len(messages)}")
                    
                elif c_mode == "auto_summarize":
                    middle_msgs = messages[k_first:-k_recent]
                    if middle_msgs:
                        print("[LlamaCPP] Запуск умного сжатия контекста (auto_summarize)...")
                        middle_text = ""
                        for m in middle_msgs:
                            role = m.get("role", "unknown").upper()
                            content = m.get("content", "")
                            if isinstance(content, list):
                                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                                content_str = " ".join(text_parts).strip()
                            else:
                                content_str = str(content).strip()
                            middle_text += f"[{role}]: {content_str}\n\n"

                        sum_payload = {
                            "messages": [
                                {"role": "system", "content": chat_config.get("summarize_prompt", "Summarize the conversation.")},
                                {"role": "user", "content": middle_text}
                            ],
                            "temperature": 0.3,
                            "max_tokens": chat_config.get("summary_max_tokens", 500),
                            "stream": False
                        }
                        try:
                            sum_req = urllib.request.Request(f"http://127.0.0.1:{ACTIVE_SERVERS[server_id]['port']}/v1/chat/completions", data=json.dumps(sum_payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
                            with urllib.request.urlopen(sum_req) as sum_res:
                                sum_data = json.loads(sum_res.read().decode('utf-8'))
                                summary_text = sum_data["choices"][0]["message"]["content"].strip()
                                messages = messages[:k_first] + [{"role": "system", "content": f"[SYSTEM NOTE: ARCHIVED PREVIOUS EVENTS]\n{summary_text}"}] + messages[-k_recent:]
                                print("[LlamaCPP] Сжатие успешно завершено.")
                        except Exception as e:
                            print(f"[LlamaCPP Warning] Ошибка при сжатии контекста: {e}")
        else:
            messages = []
            sys_str = ""
            if system_prompt_preset != NO_SYSTEM_PROMPT:
                # Пресет выбран → используем ТОЛЬКО его, system_prompt_text игнорируется
                preset_path = folder_paths.get_full_path(PROMPT_FOLDER, system_prompt_preset)
                if preset_path and os.path.exists(preset_path):
                    with open(preset_path, "r", encoding="utf-8") as f:
                        sys_str = f.read().strip()
            else:
                # Пресет не выбран → используем ручной system_prompt_text
                if system_prompt_text.strip():
                    sys_str = system_prompt_text.strip()

            if sys_str.strip():
                messages.append({"role": "system", "content": sys_str.strip()})

        #  Определяем, является ли это продолжением диалога, строго ДО усечения истории
        is_followup = False
        if chat_history is not None:
            orig_history = chat_history.get("messages", [])
            is_followup = any(m.get("role") == "user" for m in orig_history)

        # Добавление реплики пользователя
        user_content = []
        
        # Чтобы модель не "сбрасывала" контекст и не описывала картинку заново с нуля,
        # мы прикрепляем медиа ТОЛЬКО к самому первому сообщению в сессии.
        # Источники: images (autogrow), videos (autogrow), audios (autogrow).
        def _sorted_autogrow(d):
            if not d:
                return []
            def _key(k):
                if "_" in k:
                    _, _, n = k.rpartition("_")
                    if n.isdigit():
                        return (0, int(n))
                return (1, str(k))
            return [d[k] for k in sorted(d.keys(), key=_key)]

        image_list = _sorted_autogrow(images) if images else []
        video_list = _sorted_autogrow(videos) if videos else []
        audio_list = _sorted_autogrow(audios) if audios else []

        if not is_followup and (image_list or video_list or audio_list):
            if not mm_path:
                print("[LlamaCPP Warning] Переданы медиа (images/videos/audios), но mmproj не выбран! Все медиа будут проигнорированы.")
            else:
                # === ВАРИАНТ A: сквозная нумерация + явные лейблы ===
                # Images: каждый кадр батча раскрывается как отдельный Image N (сквозной индекс).
                #         Модель может ссылаться на любой кадр по номеру: "use Image 3 as reference".
                # Videos: каждый пин = Video N, внутри которого перечислены последовательные кадры.
                #         Модель понимает, что это одно видео, и может ссылаться "Video 2 frames".
                # Audios: каждый пин = Audio N.
                #
                # Итоговый формат (пример):
                #   [Image 1] \n [jpeg] \n [Image 2] \n [jpeg] ...
                #   [Video 1 — 8 sequential frames] \n [frame 1 jpeg] \n [frame 2 jpeg] ...
                #   [Audio 1] \n [wav]

                # --- IMAGES: сквозная нумерация всех картинок из всех пинов ---
                img_counter = 0
                for tensor in image_list:
                    if tensor is None:
                        continue
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    b64_images = tensors_to_base64_list(tensor)
                    for b64_img in b64_images:
                        img_counter += 1
                        user_content.append({"type": "text", "text": f"[Image {img_counter}]"})
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

                # --- VIDEOS: каждый пин — отдельное Video N, кадры внутри ---
                for vid_idx, tensor in enumerate(video_list):
                    if tensor is None:
                        continue
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    # Оригинальное поведение: tensors_to_base64_list сама делает равномерную
                    # выборку через np.linspace(0, total-1, max_frames, dtype=int)
                    b64_video_frames = tensors_to_base64_list(tensor, max_frames=max_video_frames)
                    user_content.append({
                        "type": "text",
                        "text": f"[Video {vid_idx+1} — {len(b64_video_frames)} sequential frames]"
                    })
                    for b64_frame in b64_video_frames:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}})

                # --- AUDIOS: каждый пин — отдельное Audio N ---
                for aud_idx, audio in enumerate(audio_list):
                    if audio is None:
                        continue
                    try:
                        audio_b64 = audio_to_base64_wav(audio)
                        user_content.append({"type": "text", "text": f"[Audio {aud_idx+1}]"})
                        user_content.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav"
                            }
                        })
                        print(f"[LlamaCPP] Audio {aud_idx+1} processed.")
                    except Exception as e:
                        print(f"[LlamaCPP Warning] Не удалось обработать audio {aud_idx+1}: {e}")

        user_content.append({"type": "text", "text": prompt})

        # Фикс: если сообщение содержит только текст (уточнение),
        # передаем его простой строкой. Многие модели сходят с ума и забывают, 
        # что это диалог, если история состоит из сложных multi-modal словарей.
        if len(user_content) == 1 and user_content[0].get("type") == "text":
            user_content = user_content[0]["text"]

        messages.append({"role": "user", "content": user_content})

        # ИЗВЛЕКАЕМ ДОПОЛНИТЕЛЬНЫЕ СЕМПЛЕРЫ
        samplers_dict = extra_samplers if extra_samplers else {}
        banned_tokens_str = samplers_dict.get("banned_tokens", "")
        
        payload = {
            "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "top_k": top_k, "top_p": top_p, "seed": normalize_llama_seed(seed),
            "stream": True,
            "stream_options": {"include_usage": True},
            "dynatemp_range": samplers_dict.get("dynatemp_range", 0.0),
            "dynatemp_exponent": samplers_dict.get("dynatemp_exponent", 1.0),
            "xtc_probability": samplers_dict.get("xtc_probability", 0.0),
            "xtc_threshold": samplers_dict.get("xtc_threshold", 0.1),
            "smoothing_factor": samplers_dict.get("smoothing_factor", 0.0),
            "smoothing_curve": samplers_dict.get("smoothing_curve", 1.0),
            "min_p": samplers_dict.get("min_p", 0.05),
            "presence_penalty": samplers_dict.get("presence_penalty", 0.0),
            "frequency_penalty": samplers_dict.get("frequency_penalty", 0.0),
            "repeat_penalty": samplers_dict.get("repeat_penalty", 1.1),
            "mirostat": samplers_dict.get("mirostat", 0),
            "mirostat_tau": samplers_dict.get("mirostat_tau", 5.0),
            "mirostat_eta": samplers_dict.get("mirostat_eta", 0.1),
            "tfs_z": samplers_dict.get("tfs_z", 1.0),
            "typical_p": samplers_dict.get("typical_p", 1.0),
        }

        if reasoning_budget > 0:
            payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_budget": reasoning_budget
            }
            if reasoning_budget_message.strip():
                payload["chat_template_kwargs"]["reasoning_budget_message"] = reasoning_budget_message.strip()

        # ОБРАБОТКА ЗАПРЕЩЕННЫХ СЛОВ (LOGIT BIAS)
        if banned_tokens_str and banned_tokens_str.strip():
            logit_bias = {}
            for word in banned_tokens_str.split('\n'):
                word = word.strip()
                if not word: continue
                
                tok_url = f"http://127.0.0.1:{ACTIVE_SERVERS[server_id]['port']}/tokenize"
                tok_data = json.dumps({"content": word}).encode('utf-8')
                tok_req = urllib.request.Request(tok_url, data=tok_data, headers={'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(tok_req) as response:
                        tok_res = json.loads(response.read().decode('utf-8'))
                        # Выставляем вероятность этих токенов на -100.0
                        for tid in tok_res.get("tokens", []):
                            logit_bias[str(tid)] = -100.0
                except Exception as e:
                    print(f"[LlamaCPP Warning] Не удалось токенизировать запрещенное слово '{word}': {e}")
            
            if logit_bias:
                payload["logit_bias"] = logit_bias

        # SEND REQUEST
        url = f"http://127.0.0.1:{ACTIVE_SERVERS[server_id]['port']}/v1/chat/completions"
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        clean_text = ""
        thoughts_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        t_prompt = 0.0
        t_gen = 0.0

        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if comfy.model_management.processing_interrupted():
                        print(f"\n[LlamaCPP] Генерация на сервере '{server_id}' отменена пользователем (Interrupt).")
                        clean_text += "\n\n[Генерация прервана]"
                        break
                        
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line.startswith("data: "): continue
                    
                    content = decoded_line[6:]
                    if content == "[DONE]": break
                    
                    try:
                        chunk = json.loads(content)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            clean_text += delta.get("content", "") or ""
                            thoughts_text += delta.get("reasoning_content", "") or ""
                        
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)

                        if "timings" in chunk:
                            t_prompt = chunk["timings"].get("prompt_per_second", 0)
                            t_gen = chunk["timings"].get("predicted_per_second", 0)
                            
                    except json.JSONDecodeError: pass

        except urllib.error.HTTPError as e:
            clean_text = f"API Ошибка {e.code}: {e.read().decode('utf-8')}"
        except Exception as e:
            clean_text = f"Сетевая ошибка: {e}"

        # Форматируем итоговые строки метрик на основе накопленных за время стрима данных
        ctx_total = prompt_tokens + completion_tokens
        if ctx_total > 0:
            ctx_pct = (ctx_total / ctx_size * 100) if ctx_size > 0 else 0
            usage_stats = f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Context: {ctx_total} / {ctx_size} ({ctx_pct:.1f}%)"
            perf_text = f"Ctx: {ctx_total}/{ctx_size} ({ctx_pct:.1f}%) | P: {t_prompt:.1f} t/s | G: {t_gen:.1f} t/s"
        else:
            usage_stats = "No usage data"
            perf_text = f"P: {t_prompt:.1f} t/s | G: {t_gen:.1f} t/s"

        if "<think>" in clean_text:
            think_match = re.search(r'<think>(.*?)</think>', clean_text, flags=re.DOTALL | re.IGNORECASE)
            if think_match:
                if not thoughts_text: thoughts_text = think_match.group(1).strip()
                clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
            else:
                think_match_open = re.search(r'<think>(.*)', clean_text, flags=re.DOTALL | re.IGNORECASE)
                if think_match_open:
                    if not thoughts_text: thoughts_text = think_match_open.group(1).strip()
                    clean_text = re.sub(r'<think>.*', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
                    clean_text = "[ОШИБКА: Модели не хватило max_tokens для ответа.]\n" + clean_text

        # СОХРАНЕНИЕ В ИСТОРИЮ (С учетом усечений)
        if chat_history is not None:
            s_id = chat_history.get("session_id", "default")
            global CHAT_SESSIONS
            if s_id in CHAT_SESSIONS:
                # В messages уже лежит реплика пользователя, добавляем только ответ модели
                CHAT_SESSIONS[s_id] = messages + [{"role": "assistant", "content": clean_text.strip()}]
            
            out_history = {"session_id": s_id, "messages": CHAT_SESSIONS[s_id], "config": chat_config}
        else:
            out_history = {
                "session_id": "single_turn", 
                "messages": messages + [{"role": "assistant", "content": clean_text.strip()}]
            }

        if not keep_model_loaded:
            kill_server(server_id)

        return comfy_io.NodeOutput(clean_text.strip(), thoughts_text.strip(), perf_text, usage_stats, out_history)


class LlamaCPPUnloadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "any_input": (ANY, {"tooltip": "Подключите сюда что угодно, чтобы заставить эту ноду дождаться выполнения предыдущих шагов перед выгрузкой."})
            },
            "required": {
                "unload_active": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "Unload", 
                    "label_off": "Pass only",
                    "tooltip": "Выгрузка активна. Включите, чтобы принудительно убить сервер Llama и освободить видеопамять (VRAM)."
                }),
                "server_id": ("STRING", {
                    "default": "all",
                    "tooltip": "Идентификатор сервера для выгрузки. Укажите 'all' чтобы выгрузить все запущенные серверы."
                }),
                "clear_all_chats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Очистить историю абсолютно всех чат-сессий из оперативной памяти."
                })
            }
        }

    RETURN_TYPES = (ANY,)
    FUNCTION = "unload_models"
    CATEGORY = "LlamaCPP/Memory"
    OUTPUT_NODE = True

    def unload_models(self, unload_active, server_id="all", clear_all_chats=False, any_input=None):
        global CHAT_SESSIONS
        if clear_all_chats:
            CHAT_SESSIONS.clear()
            print("[LlamaCPP] Все сессии чатов успешно очищены из памяти.")
            
        if unload_active:
            if server_id == "all" or not server_id.strip():
                print("\n[LlamaCPP] Нода инициировала выгрузку всех серверов...")
                kill_all_servers()
            else:
                print(f"\n[LlamaCPP] Нода инициировала выгрузку сервера '{server_id}'...")
                kill_server(server_id)
        return (any_input, )

# =======================================================================
# 5. РЕГИСТРАЦИЯ КЛАССОВ И ИМЕН В COMFYUI
# =======================================================================

NODE_CLASS_MAPPINGS = {
    "LlamaCPP_AdvancedSamplers": LlamaCPPAdvancedSamplersNode,
    "LlamaCPP_Subprocess": LlamaCPPSubprocessNode,
    "LlamaCPP_UnloadAll": LlamaCPPUnloadNode,
    "LlamaCPP_ChatHistory": LlamaCPPChatHistoryNode,
    "LlamaCPP_FormatHistory": LlamaCPPFormatHistoryNode,
    "LlamaCPP_SpeculativeSettings": LlamaCPPSpeculativeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPP_AdvancedSamplers": "LlamaCPP Advanced Samplers",
    "LlamaCPP_Subprocess": "LlamaCPP Server Model",
    "LlamaCPP_UnloadAll": "LlamaCPP Unload All",
    "LlamaCPP_ChatHistory": "LlamaCPP Chat History",
    "LlamaCPP_FormatHistory": "LlamaCPP Format History",
    "LlamaCPP_SpeculativeSettings": "LlamaCPP Speculative Settings"
}