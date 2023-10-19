@echo on
call python -m venv venv
call venv\scripts\activate
mkdir base-model-path
mkdir base-model-path-sdxl
mkdir class-imgs-db
mkdir instance-imgs
mkdir instance-imgs-sdxl
mkdir output-path
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install bitsandbytes_windows
