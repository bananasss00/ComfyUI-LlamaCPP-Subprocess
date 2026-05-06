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

def llm_root() -> Path:
    return Path(folder_paths.models_dir) / "LLM"

def prompt_root() -> Path:
    return llm_root() / "prompts"

def register_folders() -> None:
    llm_dir = llm_root()
    prompts_dir = prompt_root()
    llm_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
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

LLAMA_CPP_RELEASE_TAG = "b9041"
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
            matches = [a for a in release.get("assets", []) if fnmatch.fnmatch(a.get("name", "").lower(), pattern.lower())]
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
# 3. ОСНОВНАЯ ЛОГИКА И СЕРВЕР
# =======================================================================

ACTIVE_SERVER = {}
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

def kill_active_server():
    global ACTIVE_SERVER
    if "process" in ACTIVE_SERVER and ACTIVE_SERVER["process"]:
        try:
            ACTIVE_SERVER["process"].kill()
            ACTIVE_SERVER["process"].wait(timeout=5)
        except: pass
    if "log_file" in ACTIVE_SERVER and ACTIVE_SERVER["log_file"]:
        try: ACTIVE_SERVER["log_file"].close()
        except: pass
    ACTIVE_SERVER = {}
    print("[LlamaCPP] Модель выгружена, процесс завершен.")

class LlamaCPPSubprocessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_options(), {
                    "tooltip": "GGUF model. Place files in ComfyUI/models/LLM. mmproj files are hidden from this list."
                }),
                "mmproj": (mmproj_options(), {
                    "default": NO_MMPROJ, 
                    "tooltip": "Vision projector GGUF. Required for images/video. Place files in ComfyUI/models/LLM."
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Опиши это видео или изображения подробно.",
                    "tooltip": "User prompt sent to the selected model."
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 1, "max": 32768,
                    "tooltip": "Maximum number of tokens to generate."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature. Lower is more deterministic."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling threshold."
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 1, "max": 100,
                    "tooltip": "Top-K sampling cutoff."
                }),
                "ctx_size": ("INT", {
                    "default": 16384, "min": 512, "max": 128000, "step": 256,
                    "tooltip": "Context window size in tokens. Larger context uses more VRAM."
                }),
                "memory_mode": (["auto", "gpu_layers", "cpu_moe_layers", "gpu_and_cpu_moe_layers"], {
                    "default": "auto",
                    "tooltip": "Advanced memory placement mode: auto, gpu_layers, cpu_moe_layers, or gpu_and_cpu_moe_layers."
                }),
                "gpu_layers": ("INT", {
                    "default": 99, "min": -1, "max": 999, 
                    "tooltip": "Used only in gpu_layers and gpu_and_cpu_moe_layers modes. Number of model layers to place on the GPU."
                }),
                "n_cpu_moe_layers": ("INT", {
                    "default": 1, "min": 1, "max": 999, 
                    "tooltip": "Used only in cpu_moe_layers and gpu_and_cpu_moe_layers modes. Number of MoE layers to keep on the CPU."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed. Use 0 for a random seed."
                }),
                "reasoning": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Reasoning output mode."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, the server stays alive in background. If False, unloads model after generation to free VRAM."
                }),
            },
            "optional": {
                "system_prompt_preset": (system_prompt_options(), {
                    "default": NO_SYSTEM_PROMPT,
                    "tooltip": "System prompt preset. Place your .txt files in ComfyUI/models/LLM/prompts"
                }),
                "system_prompt_text": ("STRING", {
                    "multiline": True, "default": "", 
                    "tooltip": "Optional manual text prompt. Will be appended to the preset."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Optional image input. A single image or ComfyUI batch (video sequence) is passed to llama-server."
                }),
                "max_video_frames": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Maximum number of frames to sample evenly from a video/image batch."
                }),
                "audio_video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional absolute path to an external media file on disk."
                }),
                "executable_path": ("STRING", {
                    "default": "auto", 
                    "tooltip": "'auto' to auto-download, or full absolute path to your llama-server.exe"
                }),
                "extra_cli_args": ("STRING", {
                    "default": "",
                    "tooltip": "Optional advanced llama.cpp parameters. Leave empty for normal use."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thoughts", "perf")
    FUNCTION = "generate_text"
    CATEGORY = "LlamaCPP/Inference"

    def generate_text(self, model, mmproj, prompt, max_tokens, temperature, top_p, top_k, ctx_size, memory_mode, gpu_layers, 
                      n_cpu_moe_layers, seed, reasoning, keep_model_loaded, system_prompt_preset=NO_SYSTEM_PROMPT, 
                      system_prompt_text="", image=None, max_video_frames=8, audio_video_path="", executable_path="auto", extra_cli_args=""):
        
        global ACTIVE_SERVER

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
            "exe": exe_path, "model": m_path, "mmproj": mm_path, "ctx": ctx_size, 
            "mem": memory_mode, "gpu": gpu_layers, "moe": n_cpu_moe_layers,
            "args": extra_cli_args, "reasoning": reasoning
        }

        if ACTIVE_SERVER.get("config") != current_config and "process" in ACTIVE_SERVER:
            print(f"\n[LlamaCPP] Изменение настроек модели. Выгрузка старой...")
            kill_active_server()

        # START SERVER IF NEEDED
        if not ACTIVE_SERVER:
            port = get_free_port()
            cmd =[exe_path, "-m", m_path, "-c", str(ctx_size), "--port", str(port)]
            
            if memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
                cmd.extend(["-ngl", str(gpu_layers)])
            if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
                cmd.extend(["--n-cpu-moe", str(n_cpu_moe_layers)])
            
            if mm_path: cmd.extend(["--mmproj", mm_path])
            if reasoning != "auto": cmd.extend(["--reasoning", reasoning])
            if extra_cli_args: cmd.extend(extra_cli_args.split())

            print(f"\n[LlamaCPP] Запуск сервера: {' '.join(cmd)}")
            log_file_path = os.path.join(os.getcwd(), "llama_server_debug.log")
            log_file = open(log_file_path, "w", encoding="utf-8")
            
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            
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
                raise Exception(f"[LlamaCPP ERROR] Сервер не запустился. Лог: {err_text[-1000:]}")
            
            ACTIVE_SERVER = {"process": process, "config": current_config, "port": port, "log_file": log_file}
            print(f"[LlamaCPP] Сервер готов на порту {port}.")

        # COMPILE SYSTEM PROMPT
        sys_str = ""
        if system_prompt_preset != NO_SYSTEM_PROMPT:
            preset_path = folder_paths.get_full_path(PROMPT_FOLDER, system_prompt_preset)
            if preset_path and os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as f:
                    sys_str += f.read() + "\n"
        if system_prompt_text.strip():
            sys_str += system_prompt_text

        # PREPARE MESSAGES
        messages =[]
        if sys_str.strip():
            messages.append({"role": "system", "content": sys_str.strip()})

        user_content =[]
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
        messages.append({"role": "user", "content": user_content})

        payload = {
            "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "top_k": top_k, "top_p": top_p, "seed": normalize_llama_seed(seed),
            "stream": True # <-- STREAMING FOR INTERRUPTS!
        }
        
        # SEND REQUEST
        url = f"http://127.0.0.1:{ACTIVE_SERVER['port']}/v1/chat/completions"
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        clean_text = ""
        thoughts_text = ""
        perf_text = ""

        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    # ComfyUI Interrupt Support
                    if comfy.model_management.processing_interrupted():
                        print("\n[LlamaCPP] Генерация отменена пользователем (Interrupt).")
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
                        
                        if "usage" in chunk and "completion_tokens" in chunk["usage"]:
                            usage = chunk.get("usage", {})
                            # В llama-server потоке иногда пробрасывается метрика timings
                            # Для точной метрики мы можем использовать данные из лога или usage
                            pass
                            
                        # Специфичные тайминги от llama.cpp
                        if "timings" in chunk:
                            t_prompt = chunk["timings"].get("prompt_per_second", 0)
                            t_gen = chunk["timings"].get("predicted_per_second", 0)
                            perf_text = f"Prompt: {t_prompt:.1f} t/s | Generation: {t_gen:.1f} t/s"
                            
                    except json.JSONDecodeError: pass

        except urllib.error.HTTPError as e:
            clean_text = f"API Ошибка {e.code}: {e.read().decode('utf-8')}"
        except Exception as e:
            clean_text = f"Сетевая ошибка: {e}"

        # Fallback regex parsing (if model puts <think> in main content instead of reasoning_content)
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

        if not keep_model_loaded:
            kill_active_server()

        return (clean_text.strip(), thoughts_text.strip(), perf_text)


class LlamaCPPUnloadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {"any_input": (ANY, {"tooltip": "Подключите сюда что угодно (картинку, текст и т.д.)"})},
            "required": {"unload_active": ("BOOLEAN", {"default": True, "label_on": "Unload", "label_off": "Pass only"})}
        }

    RETURN_TYPES = (ANY,)
    FUNCTION = "unload_models"
    CATEGORY = "LlamaCPP/Memory"
    OUTPUT_NODE = True

    def unload_models(self, unload_active, any_input=None):
        if unload_active:
            print("\n[LlamaCPP] Нода инициировала выгрузку...")
            kill_active_server()
        return (any_input, )

NODE_CLASS_MAPPINGS = {
    "LlamaCPP_Subprocess": LlamaCPPSubprocessNode,
    "LlamaCPP_UnloadAll": LlamaCPPUnloadNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPP_Subprocess": "LlamaCPP Server Model",
    "LlamaCPP_UnloadAll": "LlamaCPP Unload All"
}