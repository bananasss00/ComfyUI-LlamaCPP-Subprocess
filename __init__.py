import os
import subprocess
import time
import urllib.request
import urllib.error
import json
import base64
import socket
import re
import numpy as np
from PIL import Image
import io
import fnmatch
import platform
import zipfile
import threading
import inspect
import atexit
import execution
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import folder_paths
import comfy.model_management

# =======================================================================
# 1. СИСТЕМА ПУТЕЙ И РЕЕСТР ФАЙЛОВ (DROPDOWNS)
# =======================================================================

LLM_FOLDER = "llm_text_processor_models"
PROMPT_FOLDER = "llm_text_processor_prompts"
NO_SYSTEM_PROMPT = "none"
NO_MMPROJ = "none"
NO_MODELS_FOUND = "No GGUF models found"

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
    files = folder_paths.get_filename_list(PROMPT_FOLDER)
    top_level_files =[name for name in files if os.sep not in name and "/" not in name]
    return [NO_SYSTEM_PROMPT] + top_level_files

def full_model_path(name: str) -> Path:
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    return Path(path) if path else Path("")

register_folders()

# =======================================================================
# 2. АВТО-СКАЧИВАНИЕ LLAMA-SERVER.EXE
# =======================================================================

LLAMA_CPP_RELEASE_TAG = "b9518"
RELEASE_API_URL = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_RELEASE_TAG}"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"

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

def ensure_llama_server_paths() -> str:
    system = platform.system().lower()
    if system != "windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Auto-download is only supported on Windows x64. Please specify your executable_path manually.")
    
    spec = WINDOWS_CUDA_13
    install_dir = VENDOR_ROOT / LLAMA_CPP_RELEASE_TAG / spec.key
    exe_path = install_dir / spec.cli_executable

    if exe_path.exists():
        return str(exe_path)

    print("[LlamaCPP] Auto-downloading llama-server.exe... Please wait.")
    install_dir.mkdir(parents=True, exist_ok=True)
    
    req = urllib.request.Request(RELEASE_API_URL, headers={"User-Agent": "ComfyUI-LLM"})
    with urllib.request.urlopen(req) as response:
        release = json.loads(response.read().decode("utf-8"))
    
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for pattern in spec.asset_patterns:
            matches =[a for a in release.get("assets", []) if fnmatch.fnmatch(a.get("name", "").lower(), pattern.lower())]
            if matches:
                asset = sorted(matches, key=lambda i: i.get("name", ""))[0]
                archive_path = temp_dir / asset["name"]
                _download_file(asset["browser_download_url"], archive_path)
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(install_dir)
                    
    if not exe_path.exists():
        raise RuntimeError(f"Failed to find {spec.cli_executable} after download.")
    return str(exe_path)

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


class LlamaCPPSubprocessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_options(), {
                    "tooltip": "Файл GGUF модели для текста. Поместите файлы в папку ComfyUI/models/LLM. Файлы mmproj (для зрения) здесь скрыты."
                }),
                "mmproj": (mmproj_options(), {
                    "default": NO_MMPROJ, 
                    "tooltip": "Файл проектора зрения (Vision GGUF). Нужен ТОЛЬКО если вы передаете картинки/видео. Обязательно должен подходить к архитектуре основной текстовой модели."
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Опиши это видео или изображения подробно.",
                    "tooltip": "Текст вашего запроса (промпт), который вы отправляете модели."
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 1, "max": 32768,
                    "tooltip": "Максимальное количество слов (токенов), которое разрешено сгенерировать модели в ответе."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Температура генерации. Низкая (0.1) делает текст строгим и предсказуемым. Высокая (1.0+) делает текст более творческим и случайным."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Top-P (Nucleus). Ограничивает выборку слов, оставляя только те, которые в сумме дают указанную вероятность (0.95 = 95% наиболее вероятных слов)."
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 1, "max": 100,
                    "tooltip": "Top-K. Жесткое ограничение: выбирать следующее слово только из K самых вероятных вариантов (например, из 40 лучших)."
                }),
                "ctx_size": ("INT", {
                    "default": 16384, "min": 512, "max": 128000, "step": 256,
                    "tooltip": "Размер контекста в токенах. Сколько истории или текста модель 'помнит'. Большие значения (32к+) сильно потребляют видеопамять (VRAM)."
                }),
                "flash_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "КРАЙНЕ РЕКОМЕНДУЕТСЯ. Flash Attention (-fa) кардинально снижает потребление видеопамяти при больших контекстах и ускоряет генерацию."
                }),
                "context_quantization": (["none", "q8", "q4"], {
                    "default": "none",
                    "tooltip": "Квантование контекста (сжатие памяти диалога). 'q8' экономит ~50% VRAM контекста, 'q4' экономит ~75%. Минимально влияет на качество."
                }),
                "memory_mode": (["auto", "gpu_layers", "cpu_moe_layers", "gpu_and_cpu_moe_layers"], {
                    "default": "auto",
                    "tooltip": "Режим распределения слоев модели (Авто, только видеокарта, частично процессор). Выбирайте 'auto', если не уверены."
                }),
                "gpu_layers": ("INT", {
                    "default": 99, "min": -1, "max": 999, 
                    "tooltip": "Сколько слоев модели загрузить в видеокарту (GPU). 999 означает 'все возможные'. Если не хватает VRAM, постепенно уменьшайте это значение."
                }),
                "n_cpu_moe_layers": ("INT", {
                    "default": 1, "min": 1, "max": 999, 
                    "tooltip": "Количество MoE-слоев, которые останутся на процессоре (только для моделей со смешанной архитектурой)."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Зерно случайности. Постоянное значение (например 42) даст одинаковый текст при тех же настройках. 0 = случайный текст каждый раз."
                }),
                "reasoning": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Режим размышления (Reasoning). Нужен для умных моделей вроде DeepSeek-R1, чтобы они выводили процесс решения задачи."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Если включено, сервер останется висеть в памяти после ответа (для быстрых повторных запросов). Если выключено - модель полностью выгрузится, освободив VRAM."
                }),
                "batch_size": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 64,
                    "tooltip": "Скорость чтения промпта (--batch-size). Сколько токенов обрабатывается за раз при анализе входа. 512 - баланс. Уменьшите, если не хватает VRAM."
                }),
                "parallel_requests": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Количество слотов запросов (-np). Ограничение в 1 слот экономит VRAM, не позволяя серверу резервировать память под параллельных пользователей."
                }),
                "no_mmap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Отключить MMap. Загрузит модель в ОЗУ напрямую, игнорируя дисковый кэш ОС. Помогает при некоторых ошибках чтения или на медленных дисках."
                }),
                "no_warmup": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Отключить 'разогрев'. Пропускает начальный тестовый прогон матрицы при старте сервера, экономя пару секунд времени запуска."
                }),
                "mlock": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Блокировка в ОЗУ (--mlock). Запрещает системе выгружать модель из оперативной памяти в файл подкачки диска. Устраняет фризы, но требует много свободной ОЗУ."
                }),
                "fit_target_mib": ("INT", {
                    "default": 0, "min": 0, "max": 24000, "step": 256,
                    "tooltip": "Гарантированный запас свободной VRAM (-fitt) в Мегабайтах. Укажите, сколько видеопамяти Llama обязана оставить нетронутой (например, 3096 для 3 ГБ под генерацию картинок)."
                }),
            },
            "optional": {
                "override_tensor": ("STRING", {
                    "default": "",
                    "tooltip": "Хак для видеопамяти (-ot). Например, '.*ffn_down.*=CPU' принудительно перенесет самые тяжелые слои модели на процессор, освободив VRAM."
                }),
                "extra_samplers": ("LLAMA_SAMPLERS", {
                    "tooltip": "Подключите сюда выход ноды LlamaCPP Advanced Samplers для тонкой настройки вероятностей."
                }),
                "chat_history": ("LLAMA_CHAT_HISTORY", {
                    "tooltip": "Подключите историю сессии чата, чтобы включить режим памяти и вести диалог с моделью."
                }),
                "system_prompt_preset": (system_prompt_options(), {
                    "default": NO_SYSTEM_PROMPT,
                    "tooltip": "Шаблон системного промпта. Ваши .txt файлы из папки ComfyUI/models/LLM/prompts появятся в этом списке."
                }),
                "system_prompt_text": ("STRING", {
                    "multiline": True, "default": "", 
                    "tooltip": "Дополнительный ручной текст системного промпта. Будет приклеен к выбранному выше шаблону."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Подключите сюда картинку или батч картинок (видео). Обязательно выберите mmproj-файл выше!"
                }),
                "max_video_frames": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Максимальное количество кадров, которое будет равномерно извлечено из батча картинок (видео) и отправлено модели."
                }),
                "audio_video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Абсолютный путь к медиафайлу на вашем компьютере (опционально, вместо передачи через вход image)."
                }),
                "executable_path": ("STRING", {
                    "default": "auto", 
                    "tooltip": "Путь до исполняемого файла llama-server.exe. Значение 'auto' автоматически скачает и настроит нужную версию."
                }),
                "extra_cli_args": ("STRING", {
                    "default": "",
                    "tooltip": "Любые другие аргументы командной строки llama.cpp. Только для продвинутых юзеров."
                }),
                "extra_reserve_vram": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "Дополнительный 'виртуальный' резерв памяти (в ГБ), который мы сообщаем ComfyUI. Помогает избежать ошибок Out Of Memory (OOM) во время генерации картинок."
                }),
                "server_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Уникальный идентификатор сервера. Позволяет одновременно запускать несколько разных моделей Llama."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "LLAMA_CHAT_HISTORY")
    RETURN_NAMES = ("text", "thoughts", "perf", "usage_stats", "chat_history")
    FUNCTION = "generate_text"
    CATEGORY = "LlamaCPP/Inference"

    def generate_text(self, server_id, model, mmproj, prompt, max_tokens, temperature, top_p, top_k, ctx_size, flash_attention, context_quantization, memory_mode, gpu_layers, 
                      n_cpu_moe_layers, seed, reasoning, keep_model_loaded, batch_size=512, parallel_requests=1, no_mmap=False, no_warmup=False, mlock=False, fit_target_mib=0, system_prompt_preset=NO_SYSTEM_PROMPT, 
                      system_prompt_text="", image=None, max_video_frames=8, audio_video_path="", executable_path="auto", extra_cli_args="", override_tensor="", extra_samplers=None, extra_reserve_vram=0.6,
                      chat_history=None):
        
        global ACTIVE_SERVERS, ORIGINAL_EXTRA_RESERVED_VRAM

        if model == NO_MODELS_FOUND:
            raise ValueError("No models found. Please put .gguf files in ComfyUI/models/LLM")

        # Resolve paths
        m_path = str(full_model_path(model))
        mm_path = str(full_model_path(mmproj)) if mmproj != NO_MMPROJ else ""
        
        if executable_path.strip().lower() == "auto":
            exe_path = ensure_llama_server_paths()
        else:
            exe_path = executable_path

        # Create config hash to detect if we need to restart server
        current_config = {
            "exe": exe_path, "model": m_path, "mmproj": mm_path, "ctx": ctx_size, "ctx_q": context_quantization,
            "mem": memory_mode, "gpu": gpu_layers, "moe": n_cpu_moe_layers,
            "args": extra_cli_args, "reasoning": reasoning, "flash_attn": flash_attention,
            "batch": batch_size, "np": parallel_requests, "no_mmap": no_mmap, "no_warmup": no_warmup, 
            "mlock": mlock, "fitt": fit_target_mib, "ot": override_tensor
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
            
            if context_quantization == "q8":
                cmd.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
            elif context_quantization == "q4":
                cmd.extend(["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])
            
            if mm_path: cmd.extend(["--mmproj", mm_path])
            if reasoning != "auto": cmd.extend(["--reasoning", reasoning])
            if extra_cli_args: cmd.extend(extra_cli_args.split())

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
                preset_path = folder_paths.get_full_path(PROMPT_FOLDER, system_prompt_preset)
                if preset_path and os.path.exists(preset_path):
                    with open(preset_path, "r", encoding="utf-8") as f:
                        sys_str += f.read() + "\n"
            if system_prompt_text.strip():
                sys_str += system_prompt_text

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
        # мы прикрепляем картинки и видео ТОЛЬКО к самому первому сообщению в сессии.
        if not is_followup:
            if audio_video_path and os.path.exists(audio_video_path):
                user_content.append({"type": "text", "text": f"Media file attached: {audio_video_path}\n"})
            
            if image is not None:
                if not mm_path:
                    print("[LlamaCPP Warning] Передано изображение/видео, но mmproj не выбран! Изображение будет проигнорировано.")
                else:
                    b64_images = tensors_to_base64_list(image, max_frames=max_video_frames)
                    if len(b64_images) > 1:
                        user_content.append({"type": "text", "text": "(Video sequence frames attached)\n"})
                    for b64_img in b64_images:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

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

        return (clean_text.strip(), thoughts_text.strip(), perf_text, usage_stats, out_history)


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
                })
            }
        }

    RETURN_TYPES = (ANY,)
    FUNCTION = "unload_models"
    CATEGORY = "LlamaCPP/Memory"
    OUTPUT_NODE = True

    def unload_models(self, unload_active, server_id="all", any_input=None):
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
    "LlamaCPP_FormatHistory": LlamaCPPFormatHistoryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPP_AdvancedSamplers": "LlamaCPP Advanced Samplers",
    "LlamaCPP_Subprocess": "LlamaCPP Server Model",
    "LlamaCPP_UnloadAll": "LlamaCPP Unload All",
    "LlamaCPP_ChatHistory": "LlamaCPP Chat History",
    "LlamaCPP_FormatHistory": "LlamaCPP Format History"
}