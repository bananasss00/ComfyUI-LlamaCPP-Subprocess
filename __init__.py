import os
import subprocess
import time
import urllib.request
import json
import base64
import socket
import torch
import numpy as np
from PIL import Image
import io

# Глобальный словарь для умного управления процессами сервера
# Формат: {"process": Popen_object, "model_path": str, "port": int}
ACTIVE_SERVER = {}

def get_free_port():
    """Находит свободный порт в системе для запуска сервера."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def tensor_to_base64(image_tensor):
    """Конвертирует изображение ComfyUI (тензор) в Base64 для отправки в llama.cpp"""
    i = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def kill_active_server():
    """Принудительно убивает текущий процесс llama-server для очистки памяти."""
    global ACTIVE_SERVER
    if "process" in ACTIVE_SERVER and ACTIVE_SERVER["process"] is not None:
        try:
            ACTIVE_SERVER["process"].kill()
            ACTIVE_SERVER["process"].wait(timeout=5)
        except Exception as e:
            print(f"[LlamaCPP] Ошибка при выгрузке модели: {e}")
    ACTIVE_SERVER = {}
    print("[LlamaCPP] Модель успешно выгружена из памяти.")

class LlamaCPPSubprocessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Важно: указываем именно llama-server.exe
                "executable_path": ("STRING", {"default": r"D:\Programs\Portable\llama.cpp\llama-server.exe"}),
                "model_path": ("STRING", {"default": r"D:\Models\llama-3.gguf"}),
                "prompt": ("STRING", {"multiline": True, "default": "Опиши это изображение или ответь на вопрос."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "ctx_size": ("INT", {"default": 2048, "min": 512, "max": 128000, "step": 256}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "gpu_layers": ("INT", {"default": 99, "min": -1, "max": 100}), # -1 для авто/без GPU, 99 для фулл GPU
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant."}),
                "image": ("IMAGE", ), # Картинка из ComfyUI
                "audio_video_path": ("STRING", {"default": ""}), # Путь к файлу (если модель поддерживает видео/аудио)
                "mmproj_path": ("STRING", {"default": ""}), # Путь к mmproj для LLaVA
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "extra_cli_args": ("STRING", {"default": ""}), # Любые дополнительные параметры для llama.cpp
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "LlamaCPP/Inference"

    def generate_text(self, executable_path, model_path, prompt, keep_model_loaded, ctx_size, max_tokens, 
                      temperature, gpu_layers, system_prompt="", image=None, audio_video_path="", 
                      mmproj_path="", top_k=40, top_p=0.95, extra_cli_args=""):
        
        global ACTIVE_SERVER

        # Умное управление памятью: если запрашивается другая модель, убиваем старую
        if ACTIVE_SERVER.get("model_path") != model_path and "process" in ACTIVE_SERVER:
            print(f"[LlamaCPP] Смена модели обнаружена. Выгрузка старой модели...")
            kill_active_server()

        # Запуск сервера, если он еще не запущен
        if not ACTIVE_SERVER:
            port = get_free_port()
            cmd = [
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

            print(f"[LlamaCPP] Запуск сервера: {' '.join(cmd)}")
            # Запускаем в фоне, подавляя логи (чтобы не спамить консоль ComfyUI)
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # Ждем пока сервер поднимется
            server_ready = False
            for _ in range(60): # Ждем до 60 секунд
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
                raise Exception("[LlamaCPP] Сервер не смог запуститься или время ожидания истекло.")
            
            ACTIVE_SERVER = {"process": process, "model_path": model_path, "port": port}
            print(f"[LlamaCPP] Модель {os.path.basename(model_path)} успешно загружена.")

        # Формируем payload для генерации (совместим с OpenAI API, который поддерживает llama-server)
        port = ACTIVE_SERVER["port"]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Обработка пользовательского промпта и мультимодальности
        user_content = []
        
        # Если есть аудио/видео путь, добавляем его в текст (зависит от вашей модели Qwen-Audio и т.д.)
        if audio_video_path and os.path.exists(audio_video_path):
            user_content.append({"type": "text", "text": f"Media file attached: {audio_video_path}\n"})
            # Если ваша сборка llama.cpp имеет кастомные теги для аудио:
            user_content.append({"type": "text", "text": f"<|audio|>{audio_video_path}<|endofaudio|>\n"})
        
        user_content.append({"type": "text", "text": prompt})

        # Если подана картинка из ComfyUI
        if image is not None:
            base64_img = tensor_to_base64(image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

        messages.append({"role": "user", "content": user_content})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p
        }

        # Отправляем запрос на генерацию
        print("[LlamaCPP] Генерация ответа...")
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                generated_text = result['choices'][0]['message']['content']
        except Exception as e:
            generated_text = f"Ошибка генерации: {e}"

        # Управление выгрузкой
        if not keep_model_loaded:
            print("[LlamaCPP] keep_model_loaded=False. Выгружаем модель...")
            kill_active_server()

        return (generated_text, )


class LlamaCPPUnloadNode:
    """Нода для ручной принудительной очистки VRAM (убивает процесс llama-server)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Dummy вход, чтобы можно было прикрепить ноду к графу выполнения
                "trigger": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload_models"
    CATEGORY = "LlamaCPP/Memory"

    def unload_models(self, trigger):
        kill_active_server()
        return ("All Llama models unloaded from VRAM.", )

# Регистрация нод в ComfyUI
NODE_CLASS_MAPPINGS = {
    "LlamaCPP_Subprocess": LlamaCPPSubprocessNode,
    "LlamaCPP_UnloadAll": LlamaCPPUnloadNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPP_Subprocess": "LlamaCPP Process (Multimodal)",
    "LlamaCPP_UnloadAll": "LlamaCPP Unload All Models"
}