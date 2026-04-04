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

ACTIVE_SERVER = {}

class AnyType(str):
    """Специальный класс-хак для поддержки любого типа данных (ANY) на входах ComfyUI"""
    def __ne__(self, __value: object) -> bool:
        return False

ANY = AnyType("*")

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def tensors_to_base64_list(image_tensor, max_frames=8):
    """
    Конвертирует батч картинок (видео) в список строк Base64.
    Равномерно сэмплирует max_frames, если кадров слишком много.
    """
    total_frames = image_tensor.shape[0]
    
    # Если кадров больше, чем разрешено, берем равномерные "срезы"
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        
    print(f"\n[LlamaCPP DEBUG] Получено видео/батч из {total_frames} кадров. Сэмплируем {len(indices)} кадров: {indices}")

    b64_list =[]
    for i in indices:
        img_np = 255. * image_tensor[i].cpu().numpy()
        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        
        # Можно раскомментировать, чтобы посмотреть размеры каждого кадра
        # print(f"[LlamaCPP DEBUG] Кадр {i}: Разрешение {img.width}x{img.height}")
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        b64_list.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        
    return b64_list

def kill_active_server():
    global ACTIVE_SERVER
    if "process" in ACTIVE_SERVER and ACTIVE_SERVER["process"] is not None:
        try:
            ACTIVE_SERVER["process"].kill()
            ACTIVE_SERVER["process"].wait(timeout=5)
        except Exception as e:
            print(f"[LlamaCPP ERROR] Ошибка при выгрузке модели: {e}")
    if "log_file" in ACTIVE_SERVER and ACTIVE_SERVER["log_file"] is not None:
        try:
            ACTIVE_SERVER["log_file"].close()
        except:
            pass
    ACTIVE_SERVER = {}
    print("[LlamaCPP] Модель выгружена, процесс завершен.")

class LlamaCPPSubprocessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "executable_path": ("STRING", {"default": r"D:\Programs\Portable\llama.cpp\llama-server.exe"}),
                "model_path": ("STRING", {"default": r"D:\Models\llama-3.gguf"}),
                "prompt": ("STRING", {"multiline": True, "default": "Опиши это видео или изображения подробно."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "ctx_size": ("INT", {"default": 16384, "min": 512, "max": 128000, "step": 256}), # Увеличил дефолт для видео
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "gpu_layers": ("INT", {"default": 99, "min": -1, "max": 100}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant."}),
                "image": ("IMAGE", ), # Теперь принимает и батчи (видео)
                "max_video_frames": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}), # Контроль кадров
                "audio_video_path": ("STRING", {"default": ""}), # Запасной вариант для старых моделей
                "mmproj_path": ("STRING", {"default": ""}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "extra_cli_args": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thoughts")
    FUNCTION = "generate_text"
    CATEGORY = "LlamaCPP/Inference"

    def generate_text(self, executable_path, model_path, prompt, keep_model_loaded, seed, ctx_size, max_tokens, 
                      temperature, gpu_layers, system_prompt="", image=None, max_video_frames=8, audio_video_path="", 
                      mmproj_path="", top_k=40, top_p=0.95, extra_cli_args=""):
        
        global ACTIVE_SERVER

        if ACTIVE_SERVER.get("model_path") != model_path and "process" in ACTIVE_SERVER:
            print(f"\n[LlamaCPP] Смена модели. Выгрузка старой...")
            kill_active_server()

        if not ACTIVE_SERVER:
            port = get_free_port()
            cmd =[
                executable_path,
                "-m", model_path,
                "-c", str(ctx_size),
                "--port", str(port)
            ]
            
            if gpu_layers >= 0:
                cmd.extend(["-ngl", str(gpu_layers)])
            if mmproj_path:
                cmd.extend(["--mmproj", mmproj_path])
            if extra_cli_args:
                cmd.extend(extra_cli_args.split())

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
                raise Exception("[LlamaCPP ERROR] Сервер не запустился. Смотрите llama_server_debug.log")
            
            ACTIVE_SERVER = {"process": process, "model_path": model_path, "port": port, "log_file": log_file}
            print(f"[LlamaCPP] Сервер готов на порту {port}.")

        port = ACTIVE_SERVER["port"]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        messages =[]
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content =[]
        
        # Если передан физический путь
        if audio_video_path and os.path.exists(audio_video_path):
            user_content.append({"type": "text", "text": f"Media file attached: {audio_video_path}\n"})
        
        # Обработка визуального входа ComfyUI (Картинка или Батч картинок/Видео)
        if image is not None:
            # Получаем список Base64-кадров (с лимитом max_video_frames)
            b64_images = tensors_to_base64_list(image, max_frames=max_video_frames)
            
            if len(b64_images) > 1:
                user_content.append({"type": "text", "text": "(Video sequence frames attached)\n"})
                
            # Добавляем каждый кадр в payload
            for b64_img in b64_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })

        # Добавляем текстовый промпт в самом конце массива (строго после кадров)
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        clean_text = ""
        thoughts_text = ""

        try:
            with urllib.request.urlopen(req) as response:
                raw_data = response.read().decode('utf-8')
                result = json.loads(raw_data)
                message = result['choices'][0]['message']
                
                raw_content = message.get('content', '') or ''
                api_reasoning = message.get('reasoning_content', '') or ''
                
                clean_text = raw_content
                thoughts_text = api_reasoning

                think_match = re.search(r'<think>(.*?)</think>', clean_text, flags=re.DOTALL | re.IGNORECASE)
                if think_match:
                    if not thoughts_text:
                        thoughts_text = think_match.group(1).strip()
                    clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
                else:
                    think_match_open = re.search(r'<think>(.*)', clean_text, flags=re.DOTALL | re.IGNORECASE)
                    if think_match_open:
                        if not thoughts_text:
                            thoughts_text = think_match_open.group(1).strip()
                        clean_text = re.sub(r'<think>.*', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
                        clean_text = "[ОШИБКА: Модели не хватило max_tokens, чтобы закончить мысль и выдать ответ.]\n" + clean_text

                clean_text = clean_text.lstrip() 
                
                if not clean_text.strip() and not thoughts_text.strip():
                    clean_text = "[Пустой ответ]. Возможные причины: mmproj не смог прочитать картинку/видео или закончился ctx_size."

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            clean_text = f"API Ошибка {e.code}: {error_body}"
        except Exception as e:
            clean_text = f"Сетевая ошибка: {e}"

        if not keep_model_loaded:
            kill_active_server()

        return (clean_text, thoughts_text)


class LlamaCPPUnloadNode:
    """Сквозная нода (Pass-through) для удобной выгрузки сервера прямо в рабочем процессе"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (ANY, {"tooltip": "Подключите сюда что угодно (картинку, текст и т.д.)"}),
                "unload_active": ("BOOLEAN", {"default": True, "label_on": "Unload", "label_off": "Pass only"}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "unload_models"
    CATEGORY = "LlamaCPP/Memory"

    def unload_models(self, any_input, unload_active):
        if unload_active:
            print("\n[LlamaCPP] Сквозная нода инициировала выгрузку...")
            kill_active_server()
        else:
            print("\n[LlamaCPP] Сквозная нода пропущена (unload_active = False).")
            
        return (any_input, )

NODE_CLASS_MAPPINGS = {
    "LlamaCPP_Subprocess": LlamaCPPSubprocessNode,
    "LlamaCPP_UnloadAll": LlamaCPPUnloadNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPP_Subprocess": "LlamaCPP Process (Multimodal)",
    "LlamaCPP_UnloadAll": "LlamaCPP Unload All Models"
}