## Прослушивание примеров



## Установка окружения

Зависимости разбиты на три шага из-за CUDA-сборки torch и конфликта пинов
апстрима SoloSpeech.

```powershell
# 1. Новое виртуальное окружение
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. torch/torchaudio с CUDA-сборкой — отдельно, т.к. требуют свой индекс пакетов
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Остальные прямые и обязательные транзитивные зависимости приложения
pip install -r requirements.txt

# 4. Пакет SoloSpeech без его собственных зависимостей —
#    его setup.py пинит torch==2.4.1/torchaudio==2.4.1/torchvision==0.19.1
#    и тянет тренировочный хвост (wandb, speechbrain), которые конфликтуют
#    с torch из шага 2 и не нужны нашему пайплайну
pip install --no-deps ./models/SoloSpeech
```


