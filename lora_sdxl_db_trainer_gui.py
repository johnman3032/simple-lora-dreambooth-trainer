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
    script_path = os.path.join(current_dir, "train_dreambooth_lora_sdxl.py")

    cmd = [python_exe_path, script_path]


    # Adding arguments based on the GUI values
    cmd.extend(["--pretrained_model_name_or_path", pretrained_model_entry.get()])
    cmd.extend(["--instance_data_dir", instance_data_entry.get()])
    cmd.extend(["--instance_prompt", instance_prompt_entry.get()])
    cmd.extend(["--class_prompt", class_prompt_entry.get()])
    cmd.extend(["--output_dir", output_dir_entry.get()])
    cmd.extend(["--resolution", resolution_combobox.get()])
    cmd.extend(["--train_batch_size", train_batch_size_entry.get()])
    cmd.extend(["--num_train_epochs", num_train_epochs_entry.get()])
    cmd.extend(["--gradient_accumulation_steps", gradient_accumulation_steps_entry.get()])
    cmd.extend(["--learning_rate", learning_rate_entry.get()])
    cmd.extend(["--optimizer", optimizer_combobox.get()])
    cmd.extend(["--lr_scheduler", lr_scheduler_combobox.get()])
    cmd.extend(["--lr_warmup_steps", lr_warmup_steps_entry.get()])
    cmd.extend(["--mixed_precision", mixed_precision_combobox.get()])
    cmd.extend(["--prior_generation_precision", prior_generation_precision_combobox.get()])
    cmd.extend(["--rank", rank_entry.get()])
    #cmd.extend(["--pre_compute_text_embeddings", pre_compute_text_embeddings_checkbox.get()])


    if pre_compute_text_embeddings_var.get():
        cmd.extend(["--pre_compute_text_embeddings"])
    if gradient_checkpointing_var.get():
        cmd.extend(["--gradient_checkpointing"])
    if enable_xformers_memory_var.get():
        cmd.extend(["--enable_xformers_memory_efficient_attention"])
    if scale_lr_var.get():
        cmd.extend(["--scale_lr"])
    #if set_grads_to_none_var.get():
    #    cmd.extend(["--set_grads_to_none"])

    subprocess.run(cmd)

    print(cmd)

root = tk.Tk()
root.title("SDXL LoRA Training GUI")

# Creating input widgets
pretrained_model_label = ttk.Label(root, text="Diffusers Base Model Path:")
pretrained_model_entry = ttk.Entry(root, width=50)
pretrained_model_entry.insert(0, "base-model-path-sdxl/sdxl-fp16-only")
pretrained_model_button = ttk.Button(root, text="Browse", command=lambda: select_directory(pretrained_model_entry))

instance_data_label = ttk.Label(root, text="Instance Images Directory:")
instance_data_entry = ttk.Entry(root, width=50)
instance_data_entry.insert(0, "instance-imgs-sdxl")
instance_data_button = ttk.Button(root, text="Browse", command=lambda: select_directory(instance_data_entry))

instance_prompt_label = ttk.Label(root, text="Instance Prompt (The trigger token that will \"activate\" your LoRA):")
instance_prompt_entry = ttk.Entry(root, width=50)

class_prompt_label = ttk.Label(root, text="Class Prompt (If you train images of cats, this should say \"cat\"):")
class_prompt_entry = ttk.Entry(root, width=50)

output_dir_label = ttk.Label(root, text="Output Directory (Save model here):")
output_dir_entry = ttk.Entry(root, width=50)
output_dir_entry.insert(0, "output-path")
output_dir_button = ttk.Button(root, text="Browse", command=lambda: select_directory(output_dir_entry))

resolution_label = ttk.Label(root, text="Resolution (Advanced, leave at 1024 for SDXL, choose 896 for 12GB GPU):")
resolution_combobox = ttk.Combobox(root, values=[768, 896, 1024, 1280], width=20)
resolution_combobox.set(1024)

train_batch_size_label = ttk.Label(root, text="Train Batch Size (How many images to train per step, see README for info):")
train_batch_size_entry = ttk.Entry(root, width=10)
train_batch_size_entry.insert(0, "1")

num_train_epochs_label = ttk.Label(root, text="Num of Train Epochs (see README for more info):")
num_train_epochs_entry = ttk.Entry(root, width=10)

gradient_accumulation_steps_label = ttk.Label(root, text="Gradient Accumulation Steps (Advanced, leave at 1):")
gradient_accumulation_steps_entry = ttk.Entry(root, width=10)
gradient_accumulation_steps_entry.insert(0, "1")

learning_rate_label = ttk.Label(root, text="Learning Rate (recommend 0.0001 or 2e-5 for AdamW, 1 for Prodigy):")
learning_rate_entry = ttk.Entry(root, width=15)
learning_rate_entry.insert(0, "0.0001")

optimizer_label = ttk.Label(root, text="Optimizer:")
optimizer_combobox = ttk.Combobox(root, values=["AdamW", "Prodigy"], width=20)
optimizer_combobox.set("AdamW")

lr_scheduler_label = ttk.Label(root, text="LR Scheduler:")
lr_scheduler_combobox = ttk.Combobox(root, values=["constant", "constant_with_warmup"], width=20)
lr_scheduler_combobox.set("constant")

lr_warmup_steps_label = ttk.Label(root, text="LR Warmup Steps (Only use with \"constant with warmup\", recommend 10 percent of total steps):")
lr_warmup_steps_entry = ttk.Entry(root, width=10)
lr_warmup_steps_entry.insert(0, "0")

mixed_precision_label = ttk.Label(root, text="Mixed Precision (can select bf16 if using Nvidia Ampere or newer GPU):")
mixed_precision_combobox = ttk.Combobox(root, values=["fp16", "bf16"], width=10)
mixed_precision_combobox.set("fp16")

prior_generation_precision_label = ttk.Label(root, text="Prior Generation Precision (Advanced, can select bf16 if using Nvidia Ampere or newer GPU):")
prior_generation_precision_combobox = ttk.Combobox(root, values=["no", "fp32", "fp16", "bf16"], width=10)
prior_generation_precision_combobox.set("fp16")

rank_label = ttk.Label(root, text="Network Rank, higher values for more complex image content. Increases VRAM consumption. Recommend 4-128")
rank_entry = ttk.Entry(root, width=10)
rank_entry.insert(0, "4")

pre_compute_text_embeddings_var = tk.BooleanVar()
pre_compute_text_embeddings_checkbox = ttk.Checkbutton(root, text="Pre-compute text embeddings to potentially save VRAM. Experimental.", variable=pre_compute_text_embeddings_var)

gradient_checkpointing_var = tk.BooleanVar()
gradient_checkpointing_checkbox = ttk.Checkbutton(root, text="Gradient Checkpointing (sacrifice training speed for less VRAM load)", variable=gradient_checkpointing_var)

enable_xformers_memory_var = tk.BooleanVar()
enable_xformers_memory_checkbox = ttk.Checkbutton(root, text="Enable xFormers (Recommended)", variable=enable_xformers_memory_var)

scale_lr_var = tk.BooleanVar()
scale_lr_checkbox = ttk.Checkbutton(root, text="Scale Learning Rate by grad accum steps, and batch size", variable=scale_lr_var)

#set_grads_to_none_var = tk.BooleanVar()
#set_grads_to_none_checkbox = ttk.Checkbutton(root, text="Set the gradients to none instead of zero", variable=set_grads_to_none_var)


# Button to run the script
run_button = ttk.Button(root, text="Start Training", command=run_script)

# Checkboxes
gradient_checkpointing_checkbox.grid(row=20, column=0, sticky='w', pady=5)
enable_xformers_memory_checkbox.grid(row=22, column=0, sticky='w', pady=5)
scale_lr_checkbox.grid(row=23, column=0, sticky='w', pady=5)
# Placing widgets on the window using grid layout

pretrained_model_label.grid(row=0, column=0, sticky='w', pady=5)
pretrained_model_entry.grid(row=0, column=1, pady=5)
pretrained_model_button.grid(row=0, column=2, pady=5)

instance_data_label.grid(row=2, column=0, sticky='w', pady=5)
instance_data_entry.grid(row=2, column=1, pady=5)
instance_data_button.grid(row=2, column=2, pady=5)

instance_prompt_label.grid(row=4, column=0, sticky='w', pady=5)
instance_prompt_entry.grid(row=4, column=1, pady=5)

class_prompt_label.grid(row=5, column=0, sticky='w', pady=5)
class_prompt_entry.grid(row=5, column=1, pady=5)

output_dir_label.grid(row=6, column=0, sticky='w', pady=5)
output_dir_entry.grid(row=6, column=1, pady=5)
output_dir_button.grid(row=6, column=2, pady=5)

resolution_label.grid(row=7, column=0, sticky='w', pady=5)
resolution_combobox.grid(row=7, column=1, pady=5)

train_batch_size_label.grid(row=8, column=0, sticky='w', pady=5)
train_batch_size_entry.grid(row=8, column=1, pady=5)

num_train_epochs_label.grid(row=9, column=0, sticky='w', pady=5)
num_train_epochs_entry.grid(row=9, column=1, pady=5)

gradient_accumulation_steps_label.grid(row=12, column=0, sticky='w', pady=5)
gradient_accumulation_steps_entry.grid(row=12, column=1, pady=5)

learning_rate_label.grid(row=13, column=0, sticky='w', pady=5)
learning_rate_entry.grid(row=13, column=1, pady=5)

optimizer_label.grid(row=14, column=0, sticky='w', pady=5)
optimizer_combobox.grid(row=14, column=1, pady=5)

lr_scheduler_label.grid(row=15, column=0, sticky='w', pady=5)
lr_scheduler_combobox.grid(row=15, column=1, pady=5)

lr_warmup_steps_label.grid(row=16, column=0, sticky='w', pady=5)
lr_warmup_steps_entry.grid(row=16, column=1, pady=5)

mixed_precision_label.grid(row=17, column=0, sticky='w', pady=5)
mixed_precision_combobox.grid(row=17, column=1, pady=5)

prior_generation_precision_label.grid(row=18, column=0, sticky='w', pady=5)
prior_generation_precision_combobox.grid(row=18, column=1, pady=5)

rank_label.grid(row=19, column=0, sticky='w', pady=5)
rank_entry.grid(row=19, column=1, pady=5)

pre_compute_text_embeddings_checkbox.grid(row=21, column=0, sticky='w', pady=5)

# Checkboxes
gradient_checkpointing_checkbox.grid(row=23, column=0, sticky='w', pady=5)
scale_lr_checkbox.grid(row=24, column=0, sticky='w', pady=5)
enable_xformers_memory_checkbox.grid(row=25, column=0, sticky='w', pady=5)

#set_grads_to_none_checkbox.grid(row=27, column=0, sticky='w', pady=5)

# Run button
run_button.grid(row=28, column=0, columnspan=3, pady=20)
# Run the main loop
root.mainloop()
