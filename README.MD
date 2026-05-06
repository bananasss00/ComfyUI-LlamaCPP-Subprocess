# ComfyUI LlamaCPP Server Nodes (Multimodal & Persistent)

🌍 *[Русская версия ниже](#русская-версия)*

A powerful, multimodal, and persistent integration of `llama.cpp` into ComfyUI. 

This extension runs a local `llama-server` in the background, which completely eliminates the "cold start" problem (reading huge models from disk every generation). It supports Image and Video (batch) inputs, reasoning models (DeepSeek, etc.), and advanced memory management.

🤝 **Acknowledgements:** 
This project merges our persistent server logic (video/batch support, fast API communication) with the amazing UX, auto-downloader, and advanced features from[KingManiya/ComfyUI-LLM-text-processor](https://github.com/KingManiya/ComfyUI-LLM-text-processor). Huge thanks to the author!

---

## ✨ Key Features

* 🚀 **Zero Cold Starts**: The model stays loaded in VRAM/RAM between generations. The server restarts only if you change critical settings (model, context size, memory mode).
* 🎥 **Multimodal & Video Support**: Connect a single image or an entire video (ComfyUI image batch). The node automatically samples frames evenly to fit your context!
* 🧠 **Reasoning Models Support**: Extracts `<think>` blocks and reasoning tokens into a separate output pin (`thoughts`). Perfect for DeepSeek-R1 and similar models.
* ⬇️ **Auto-Downloader**: On Windows, it automatically downloads the required `llama-server.exe` (CUDA 13) from the official GitHub. No manual setup needed!
* 📁 **Smart Dropdowns**: Automatically scans and displays your `.gguf` models and `.txt` system prompts. Separates `mmproj` files from text models for safety.
* 🛑 **Streaming & Interrupts**: Instantly stop generation by clicking `Cancel` in ComfyUI. The GPU frees up immediately.
* 🛠️ **Advanced Memory Control**: Fine-tune VRAM usage with `memory_mode`, `gpu_layers`, and specific `n_cpu_moe_layers` (essential for running heavy Mixture-of-Experts models).
* 🛡️ **Smart Error Parsing**: If you accidentally pair a text model with an incompatible `mmproj` vision encoder, the node will read the log and tell you exactly what went wrong.

---

## 📦 Installation

1. Clone this repository into your `ComfyUI/custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/bananasss00/ComfyUI-LlamaCPP-Subprocess.git
   ```
2. Restart ComfyUI.

## 📂 Folder Structure (Where to put files)

The extension registers custom folder paths in ComfyUI:

* **Models (.gguf)**: Place your LLM and Vision (`mmproj`) models here:
  `ComfyUI/models/LLM`
* **System Prompts (.txt)**: Place your text files with system instructions here:
  `ComfyUI/models/LLM/prompts`

*Note: The node automatically detects which `.gguf` files are text models and which are `mmproj` (vision projectors) based on their filenames.*

---

## 🧩 Nodes Included

### 1. LlamaCPP Server Model (Multimodal)
The main generation node.
* **Outputs**: 
  * `text`: The final model response.
  * `thoughts`: The reasoning/thinking process (if the model supports it).
  * `perf`: Generation speed metrics (Tokens/sec).
* **keep_model_loaded**: Set to `True` for fast subsequent generations. Set to `False` to kill the server and free VRAM immediately after output.

### 2. LlamaCPP Unload All
A utility node to programmatically free VRAM. Pass any flow through it, and if `unload_active` is True, it will kill the background llama.cpp server.

---
---

# Русская версия

Мощная, мультимодальная интеграция `llama.cpp` в ComfyUI с постоянным удержанием модели в памяти.

Это расширение запускает локальный `llama-server` в фоновом режиме. Это полностью решает проблему "холодного старта" (когда огромная модель каждый раз заново читается с диска). Расширение поддерживает картинки, видео (батчи кадров), рассуждающие модели (DeepSeek и др.) и продвинутое управление памятью.

🤝 **Благодарности:** 
Этот проект объединяет нашу логику постоянного сервера (поддержка видео, быстрая работа по API) с великолепным интерфейсом, авто-загрузчиком и продвинутыми функциями из [KingManiya/ComfyUI-LLM-text-processor](https://github.com/KingManiya/ComfyUI-LLM-text-processor). Огромное спасибо автору!

---

## ✨ Главные фишки

* 🚀 **Без холодных стартов**: Модель остается в видеопамяти между генерациями. Сервер перезапускается только если вы изменили важные настройки (модель, размер контекста, слои).
* 🎥 **Поддержка Видео и Картинок**: Подключайте одну картинку или целое видео (батч). Нода сама равномерно нарежет кадры (сэмплинг), чтобы уложиться в лимит.
* 🧠 **Поддержка "Думалки" (Reasoning)**: Умеет извлекать блоки `<think>` в отдельный выход (`thoughts`). Идеально для DeepSeek-R1.
* ⬇️ **Авто-скачивание**: На Windows нода сама скачает нужный `llama-server.exe` (CUDA 13) с официального GitHub. Настроил и забыл!
* 📁 **Умные списки**: Модели и текстовые промпты теперь выбираются из удобных выпадающих списков. Файлы `mmproj` автоматически отфильтровываются от текстовых моделей.
* 🛑 **Потоковая передача и Отмена**: Нажмите `Cancel` в ComfyUI, и генерация мгновенно прервется, освободив видеокарту.
* 🛠️ **Продвинутый контроль VRAM**: Настройте выгрузку слоев MoE на процессор (`n_cpu_moe_layers`), чтобы запускать тяжелые Mixture-of-Experts модели, которые не влезают в видеокарту.
* 🛡️ **Защита от ошибок**: Если вы случайно выберете несовместимые `mmproj` и текстовую модель, нода прочитает логи и выдаст понятную ошибку, а не просто зависнет.

---

## 📦 Установка

1. Склонируйте этот репозиторий в папку `ComfyUI/custom_nodes`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/bananasss00/ComfyUI-LlamaCPP-Subprocess.git
   ```
2. Перезапустите ComfyUI.

## 📂 Куда кидать файлы моделей

Расширение создает новые пути в файловой системе ComfyUI:

* **Модели (.gguf)**: Кидайте текстовые модели и Vision-энкодеры (`mmproj`) сюда:
  `ComfyUI/models/LLM`
* **Системные промпты (.txt)**: Ваши заготовки с инструкциями для ИИ кидайте сюда:
  `ComfyUI/models/LLM/prompts`

*Примечание: Нода сама поймет, какие файлы являются моделями, а какие `mmproj` (вижн-адаптерами), опираясь на их названия.*

---

## 🧩 Описание Нод

### 1. LlamaCPP Server Model (Multimodal)
Основная нода для генерации текста.
* **Выходы (Outputs)**: 
  * `text`: Финальный ответ модели.
  * `thoughts`: Мысли/Рассуждения (если модель это поддерживает).
  * `perf`: Скорость генерации (Токены в секунду).
* **keep_model_loaded**: Оставьте `True` для быстрой работы. Поставьте `False`, если хотите, чтобы сервер выключался и очищал память сразу после ответа.

### 2. LlamaCPP Unload All
Нода для принудительной выгрузки модели из видеопамяти. Пропустите через нее любую связь на холсте, и если `unload_active` включен, она убьет фоновый процесс llama.cpp.
