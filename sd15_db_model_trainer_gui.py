import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import os
import sys

def select_directory(entry):
    """Function to execute the script with the chosen arguments."""
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, directory)

def run_script():
    """Function to execute the script with the chosen arguments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe_path = os.path.join(current_dir, "venv", "Scripts", "python.exe")
    script_path = os.path.join(current_dir, "train_dreambooth.py")

    cmd = [python_exe_path, script_path]


    # Adding arguments based on the GUI values
    cmd.extend(["--pretrained_model_name_or_path", pretrained_model_entry.get()])
    cmd.extend(["--instance_data_dir", instance_data_entry.get()])
    cmd.extend(["--class_data_dir", class_data_entry.get()])
    cmd.extend(["--instance_prompt", instance_prompt_entry.get()])
    cmd.extend(["--class_prompt", class_prompt_entry.get()])
    cmd.extend(["--output_dir", output_dir_entry.get()])
    cmd.extend(["--resolution", resolution_combobox.get()])
    cmd.extend(["--train_batch_size", train_batch_size_entry.get()])
    cmd.extend(["--num_train_epochs", num_train_epochs_entry.get()])
    cmd.extend(["--checkpointing_steps", checkpointing_steps_entry.get()])
    cmd.extend(["--gradient_accumulation_steps", gradient_accumulation_steps_entry.get()])
    cmd.extend(["--learning_rate", learning_rate_entry.get()])
    cmd.extend(["--lr_scheduler", lr_scheduler_combobox.get()])
    cmd.extend(["--lr_warmup_steps", lr_warmup_steps_entry.get()])
    cmd.extend(["--mixed_precision", mixed_precision_combobox.get()])
    cmd.extend(["--prior_generation_precision", prior_generation_precision_combobox.get()])
    cmd.extend(["--offset_noise", offset_noise_entry.get()])
    # cmd.extend(["--with_prior_preservation"])
    
    # Flags
    if with_prior_preservation_var.get():
        cmd.extend(["--with_prior_preservation"])
    if set_grads_to_none_var.get():
        cmd.extend(["--set_grads_to_none"])
    if gradient_checkpointing_var.get():
        cmd.extend(["--gradient_checkpointing"])
    if use_8bit_adam_var.get():
        cmd.extend(["--use_8bit_adam"])
    if enable_xformers_memory_var.get():
        cmd.extend(["--enable_xformers_memory_efficient_attention"])
    if scale_lr_var.get():
        cmd.extend(["--scale_lr"])
    #if pre_compute_text_embeddings_var.get():
    #    cmd.extend(["--pre_compute_text_embeddings"])

    subprocess.run(cmd)

    print(cmd)

root = tk.Tk()
root.title("SD 1.5 LoRA Training GUI")

# Creating input widgets
pretrained_model_label = ttk.Label(root, text="Diffusers Base Model Path:")
pretrained_model_entry = ttk.Entry(root, width=50)
pretrained_model_entry.insert(0, "base-model-path/stable-diffusion-v1-5")
pretrained_model_button = ttk.Button(root, text="Browse", command=lambda: select_directory(pretrained_model_entry))

instance_data_label = ttk.Label(root, text="Instance Images Directory:")
instance_data_entry = ttk.Entry(root, width=50)
instance_data_entry.insert(0, "instance-imgs")
instance_data_button = ttk.Button(root, text="Browse", command=lambda: select_directory(instance_data_entry))

class_data_label = ttk.Label(root, text="Class Images Directory:")
class_data_entry = ttk.Entry(root, width=50)
class_data_entry.insert(0, "class-imgs-db")
class_data_button = ttk.Button(root, text="Browse", command=lambda: select_directory(class_data_entry))


instance_prompt_label = ttk.Label(root, text="Instance Prompt (The trigger token that will \"activate\" your LoRA, recommend choosing something random like sgu48 or catnbk):")
instance_prompt_entry = ttk.Entry(root, width=50)

class_prompt_label = ttk.Label(root, text="Class Prompt (If you train images of cats, this should say \"cat\"):")
class_prompt_entry = ttk.Entry(root, width=50)

output_dir_label = ttk.Label(root, text="Output Directory (Save model here):")
output_dir_entry = ttk.Entry(root, width=50)
output_dir_entry.insert(0, "output-path")
output_dir_button = ttk.Button(root, text="Browse", command=lambda: select_directory(output_dir_entry))

resolution_label = ttk.Label(root, text="Resolution (Advanced, leave at 512 for SD15):")
resolution_combobox = ttk.Combobox(root, values=[448, 512, 640, 768], width=20)
resolution_combobox.set(512)

train_batch_size_label = ttk.Label(root, text="Train Batch Size (How many images to train per step, value of 1 recommended):")
train_batch_size_entry = ttk.Entry(root, width=10)
train_batch_size_entry.insert(0, "1")

num_train_epochs_label = ttk.Label(root, text="Num of Train Epochs (see README for more info):")
num_train_epochs_entry = ttk.Entry(root, width=10)

checkpointing_steps_label = ttk.Label(root, text="Save checkpoint every n steps (see README for more info):")
checkpointing_steps_entry = ttk.Entry(root, width=10)

gradient_accumulation_steps_label = ttk.Label(root, text="Gradient Accumulation Steps (Advanced useage, \"1\" recommended):")
gradient_accumulation_steps_entry = ttk.Entry(root, width=10)
gradient_accumulation_steps_entry.insert(0, "1")

learning_rate_label = ttk.Label(root, text="Learning Rate (2e-6 recommended):")
learning_rate_entry = ttk.Entry(root, width=15)
learning_rate_entry.insert(0, "2e-6")

use_8bit_adam_var = tk.BooleanVar()
use_8bit_adam_checkbox = ttk.Checkbutton(root, text="Use 8-bit-AdamW (regular AdamW will be used otherwise)", variable=use_8bit_adam_var)

lr_scheduler_label = ttk.Label(root, text="LR Scheduler:")
lr_scheduler_combobox = ttk.Combobox(root, values=["constant", "constant_with_warmup"], width=20)
lr_scheduler_combobox.set("constant")

lr_warmup_steps_label = ttk.Label(root, text="LR Warmup Steps (Only use with \"constant with warmup\", recommend 10 percent of total steps, e.g. 640 for 6400):")
lr_warmup_steps_entry = ttk.Entry(root, width=10)
lr_warmup_steps_entry.insert(0, "0")

mixed_precision_label = ttk.Label(root, text="Mixed Precision (can select bf16 if your GPU supports it):")
mixed_precision_combobox = ttk.Combobox(root, values=["fp16", "bf16"], width=10)
mixed_precision_combobox.set("fp16")

prior_generation_precision_label = ttk.Label(root, text="Prior Generation Precision (Advanced, can select bf16 if your GPU supports it):")
prior_generation_precision_combobox = ttk.Combobox(root, values=["no", "fp32", "fp16", "bf16"], width=10)
prior_generation_precision_combobox.set("fp16")

offset_noise_label = ttk.Label(root, text="Offset noise, improves editability of the model. Recommended value: 0.05")
offset_noise_entry = ttk.Entry(root, width=10)
offset_noise_entry.insert(0, "4")

# Checkboxes
with_prior_preservation_var = tk.BooleanVar()
with_prior_preservation_checkbox = ttk.Checkbutton(root, text="With Prior Preservation (adds prior preservation loss, recommended.)", variable=with_prior_preservation_var)

set_grads_to_none_var = tk.BooleanVar()
set_grads_to_none_checkbox = ttk.Checkbutton(root, text="Save more VRAM by setting the grads to none instead of zero. Recommended.", variable=set_grads_to_none_var)

gradient_checkpointing_var = tk.BooleanVar()
gradient_checkpointing_checkbox = ttk.Checkbutton(root, text="Gradient Checkpointing (sacrifice training speed for less VRAM load, highly recommended)", variable=gradient_checkpointing_var)

enable_xformers_memory_var = tk.BooleanVar()
enable_xformers_memory_checkbox = ttk.Checkbutton(root, text="Enable xFormers (Recommended)", variable=enable_xformers_memory_var)

scale_lr_var = tk.BooleanVar()
scale_lr_checkbox = ttk.Checkbutton(root, text="Scale Learning Rate by batch size, choose if b.s. higher than 1", variable=scale_lr_var)

# pre_compute_text_embeddings_var = tk.BooleanVar()
# pre_compute_text_embeddings_checkbox = ttk.Checkbutton(root, text="Pre Compute Text Embeddings (Recommended, saves VRAM. Do NOT select with \"Train Text Encoder\")", variable=pre_compute_text_embeddings_var)

# Button to run the script
run_button = ttk.Button(root, text="Start Training", command=run_script)


# Checkboxes
with_prior_preservation_checkbox.grid(row=19, column=0, sticky='w', pady=5)
set_grads_to_none_checkbox.grid(row=19, column=1, sticky='w', pady=5)
gradient_checkpointing_checkbox.grid(row=20, column=0, sticky='w', pady=5)
enable_xformers_memory_checkbox.grid(row=22, column=0, sticky='w', pady=5)
scale_lr_checkbox.grid(row=23, column=0, sticky='w', pady=5)
# pre_compute_text_embeddings_checkbox.grid(row=22, column=1, sticky='w', pady=5)
# Placing widgets on the window using grid layout

pretrained_model_label.grid(row=0, column=0, sticky='w', pady=5)
pretrained_model_entry.grid(row=0, column=1, pady=5)
pretrained_model_button.grid(row=0, column=2, pady=5)

instance_data_label.grid(row=1, column=0, sticky='w', pady=5)
instance_data_entry.grid(row=1, column=1, pady=5)
instance_data_button.grid(row=1, column=2, pady=5)

class_data_label.grid(row=2, column=0, sticky='w', pady=5)
class_data_entry.grid(row=2, column=1, pady=5)
class_data_button.grid(row=2, column=2, pady=5)

instance_prompt_label.grid(row=3, column=0, sticky='w', pady=5)
instance_prompt_entry.grid(row=3, column=1, pady=5)

class_prompt_label.grid(row=4, column=0, sticky='w', pady=5)
class_prompt_entry.grid(row=4, column=1, pady=5)

output_dir_label.grid(row=5, column=0, sticky='w', pady=5)
output_dir_entry.grid(row=5, column=1, pady=5)
output_dir_button.grid(row=5, column=2, pady=5)

resolution_label.grid(row=6, column=0, sticky='w', pady=5)
resolution_combobox.grid(row=6, column=1, pady=5)

train_batch_size_label.grid(row=7, column=0, sticky='w', pady=5)
train_batch_size_entry.grid(row=7, column=1, pady=5)

num_train_epochs_label.grid(row=8, column=0, sticky='w', pady=5)
num_train_epochs_entry.grid(row=8, column=1, pady=5)

checkpointing_steps_label.grid(row=9, column=0, sticky='w', pady=5)
checkpointing_steps_entry.grid(row=9, column=1, pady=5)

gradient_accumulation_steps_label.grid(row=12, column=0, sticky='w', pady=5)
gradient_accumulation_steps_entry.grid(row=12, column=1, pady=5)

learning_rate_label.grid(row=13, column=0, sticky='w', pady=5)
learning_rate_entry.grid(row=13, column=1, pady=5)

use_8bit_adam_checkbox.grid(row=14, column=0, sticky='w', pady=5)

lr_scheduler_label.grid(row=15, column=0, sticky='w', pady=5)
lr_scheduler_combobox.grid(row=15, column=1, pady=5)

lr_warmup_steps_label.grid(row=16, column=0, sticky='w', pady=5)
lr_warmup_steps_entry.grid(row=16, column=1, pady=5)

mixed_precision_label.grid(row=17, column=0, sticky='w', pady=5)
mixed_precision_combobox.grid(row=17, column=1, pady=5)

prior_generation_precision_label.grid(row=18, column=0, sticky='w', pady=5)
prior_generation_precision_combobox.grid(row=18, column=1, pady=5)

offset_noise_label.grid(row=19, column=0, sticky='w', pady=5)
offset_noise_entry.grid(row=19, column=1, pady=5)

# Checkboxes
with_prior_preservation_checkbox.grid(row=20, column=0, sticky='w', pady=5)
set_grads_to_none_checkbox.grid(row=21, column=0, sticky='w', pady=5)
gradient_checkpointing_checkbox.grid(row=22, column=0, sticky='w', pady=5)
scale_lr_checkbox.grid(row=23, column=0, sticky='w', pady=5)
enable_xformers_memory_checkbox.grid(row=24, column=0, sticky='w', pady=5)
# pre_compute_text_embeddings_checkbox.grid(row=25, column=0, sticky='w', pady=5)

# Run button
run_button.grid(row=26, column=0, columnspan=3, pady=20)
# Run the main loop
root.mainloop()
