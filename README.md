YOLO Performance Benchmarking
Скрипт для тестирования производительности YOLO11 с различными методами обработки изображений на GPU (CUDA).

📋 Описание
Этот скрипт выполняет три различных теста для сравнения производительности модели YOLO11m при обработке изображений:

Test 1: Прямой вызов с изображением (стандартный метод)

Test 2: Обработка предобработанного тензора

Test 3: Использование torch.compile для оптимизации модели

🔧 Требования
Зависимости
pip install torch torchvision
pip install ultralytics
pip install opencv-python
Системные требования
Python 3.8+
CUDA-совместимая видеокарта NVIDIA
CUDA Toolkit (рекомендуется версия 11.8+)
PyTorch с поддержкой CUDA
📁 Необходимые файлы
Перед запуском убедитесь, что в директории проекта присутствуют:

ai_analog/
├── H100_YOLO11m.py
├── models/
│   └── yolo11m.pt          # Модель YOLO11 medium
├── bus.jpg                  # Тестовое изображение 1
└── test_image.jpg          # Тестовое изображение 2

🚀 Использование
Базовый запуск

python H100_YOLO11m.py

Пример вывода
=== Test 1: bus.jpg ===
Direct call: 0.0234 s

=== Test 2: тензор test_image.jpg ===
тензор: 0.0198 s

=== Test 3: torch.compile bus.jpg ===
compiled direct: 0.0156 s


📊 Описание тестов
Test 1: Прямой вызов модели
img = cv2.imread("bus.jpg")
img = cv2.resize(img, (640, 640))
results = model.track(img, verbose=False)
Загружает изображение с помощью OpenCV
Изменяет размер до 640x640 пикселей
Вызывает метод .track() напрямую
Базовый метод для сравнения

Test 2: Использование тензоров
img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(img_rgb).to("cuda")
tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0) / 255.0
results = model.track(tensor, verbose=False)
Конвертирует изображение из BGR в RGB
Создает PyTorch тензор и загружает на GPU
Выполняет перестановку осей (H, W, C) → (C, H, W)
Нормализует значения к диапазону [0, 1]
Может снизить накладные расходы на предобработку

Test 3: Torch Compile
model.model = torch.compile(model.model, mode="max-autotune", fullgraph=False, dynamic=False)
results = model.track(img, verbose=False)
Использует torch.compile для оптимизации графа вычислений
Режим max-autotune: максимальная оптимизация (может занять время при первом запуске)
Потенциально самый быстрый метод после компиляции

⚙️ Технические детали
Warmup
warmup_img = cv2.imread("test_image.jpg")
warmup_img = cv2.resize(warmup_img, (100, 100))
_ = model.track(warmup_img, verbose=False)
torch.cuda.synchronize()
Выполняется прогрев модели для:

Инициализации CUDA контекста
Загрузки модели в GPU память
Исключения влияния первого запуска на результаты
Синхронизация CUDA
torch.cuda.synchronize()
Используется для точного измерения времени выполнения, гарантируя завершение всех GPU операций.

Очистка памяти
torch.cuda.empty_cache()
Освобождает неиспользуемую память GPU между тестами.

