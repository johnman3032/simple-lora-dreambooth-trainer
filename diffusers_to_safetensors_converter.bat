@echo on
call venv\scripts\activate
call python convert_diffusers_to_original_stable_diffusion.py --model_path output-path --checkpoint_path converted-model-path/your_sd15_dreambooth_model.safetensors --use_safetensors --half