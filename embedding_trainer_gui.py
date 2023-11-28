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
    accelerate_exe_path = os.path.join(current_dir, "venv", "Scripts", "accelerate.exe")
    script_path = os.path.join(current_dir, "textual_inversion.py")

    cmd = [accelerate_exe_path, "launch", script_path]


    # Adding arguments based on the GUI values
    cmd.extend(["--pretrained_model_name_or_path", pretrained_model_entry.get()])
    cmd.extend(["--train_data_dir", train_data_dir_entry.get()])
    resume_checkpoint_path = resume_from_checkpoint_entry.get()
    if resume_checkpoint_path.strip():
        cmd.extend(["--resume_from_checkpoint", resume_checkpoint_path])
    cmd.extend(["--placeholder_token", placeholder_token_entry.get()])
    cmd.extend(["--initializer_token", initializer_token_entry.get()])
    cmd.extend(["--output_dir", output_dir_entry.get()])
    cmd.extend(["--resolution", resolution_combobox.get()])
    cmd.extend(["--train_batch_size", train_batch_size_entry.get()])
    cmd.extend(["--num_train_epochs", num_train_epochs_entry.get()])
    cmd.extend(["--save_steps", save_steps_entry.get()])
    cmd.extend(["--checkpointing_steps", checkpointing_steps_entry.get()])
    cmd.extend(["--gradient_accumulation_steps", gradient_accumulation_steps_entry.get()])
    cmd.extend(["--learning_rate", learning_rate_entry.get()])
    cmd.extend(["--lr_scheduler", lr_scheduler_combobox.get()])
    cmd.extend(["--lr_warmup_steps", lr_warmup_steps_entry.get()])
    cmd.extend(["--mixed_precision", mixed_precision_combobox.get()])
    cmd.extend(["--learnable_property", learnable_property_combobox.get()])
    cmd.extend(["--num_vectors", num_vectors_entry.get()])
    
    # Flags
    #if with_prior_preservation_var.get():
    #    cmd.extend(["--with_prior_preservation"])
    if center_crop_var.get():
        cmd.extend(["--center_crop"])
    if gradient_checkpointing_var.get():
        cmd.extend(["--gradient_checkpointing"])
    if use_8bit_adam_var.get():
        cmd.extend(["--use_8bit_adam"])
    if enable_xformers_memory_var.get():
        cmd.extend(["--enable_xformers_memory_efficient_attention"])
    if scale_lr_var.get():
        cmd.extend(["--scale_lr"])

    subprocess.run(cmd)

    print(cmd)

root = tk.Tk()
root.title("SD Embedding/TI Training GUI")

# Creating input widgets
pretrained_model_label = ttk.Label(root, text="Diffusers Base Model Path:")
pretrained_model_entry = ttk.Entry(root, width=50)
pretrained_model_entry.insert(0, "base-model-path/stable-diffusion-v1-5")
pretrained_model_button = ttk.Button(root, text="Browse", command=lambda: select_directory(pretrained_model_entry))

train_data_dir_label = ttk.Label(root, text="Instance Images Directory:")
train_data_dir_entry = ttk.Entry(root, width=50)
train_data_dir_entry.insert(0, "instance-imgs")
train_data_dir_button = ttk.Button(root, text="Browse", command=lambda: select_directory(train_data_dir_entry))

resume_from_checkpoint_label = ttk.Label(root, text="Path to the checkpoint folder IF you are resuming from previous training")
resume_from_checkpoint_entry = ttk.Entry(root, width=50)
resume_from_checkpoint_entry.insert(0, "")
resume_from_checkpoint_button = ttk.Button(root, text="Browse", command=lambda: select_directory(resume_from_checkpoint_entry))

placeholder_token_label = ttk.Label(root, text="A token to use as a placeholder for the concept. Take it as the equivalent to an instance token.")
placeholder_token_entry = ttk.Entry(root, width=50)

initializer_token_label = ttk.Label(root, text="A token to use as initializer word. Take it as the equivalent to a class token.")
initializer_token_entry = ttk.Entry(root, width=50)

output_dir_label = ttk.Label(root, text="Output Directory (Save embedding here):")
output_dir_entry = ttk.Entry(root, width=50)
output_dir_entry.insert(0, "output-path")
output_dir_button = ttk.Button(root, text="Browse", command=lambda: select_directory(output_dir_entry))

resolution_label = ttk.Label(root, text="Resolution (Leave it at 512 for SD1.5, set to 768 for SD2.X. If you are trying this with SDXL, set to 1024):")
resolution_combobox = ttk.Combobox(root, values=[512, 768, 896, 1024], width=20)
resolution_combobox.set(512)

train_batch_size_label = ttk.Label(root, text="Train Batch Size (How many images to train per step, value of 1 recommended):")
train_batch_size_entry = ttk.Entry(root, width=10)
train_batch_size_entry.insert(0, "1")

num_train_epochs_label = ttk.Label(root, text="Num of Train Epochs (see README for more info, same as with LoRAs):")
num_train_epochs_entry = ttk.Entry(root, width=10)

save_steps_label = ttk.Label(root, text="Save embedding model every n steps (see README for more info, same as with LoRAs):")
save_steps_entry = ttk.Entry(root, width=10)

checkpointing_steps_label = ttk.Label(root, text="Save checkpoint of the training state every n steps. This is for resuming training later on.")
checkpointing_steps_entry = ttk.Entry(root, width=10)

gradient_accumulation_steps_label = ttk.Label(root, text="Gradient Accumulation Steps (Advanced useage, \"1\" recommended):")
gradient_accumulation_steps_entry = ttk.Entry(root, width=10)
gradient_accumulation_steps_entry.insert(0, "1")

learning_rate_label = ttk.Label(root, text="Learning Rate (0.0001 recommended):")
learning_rate_entry = ttk.Entry(root, width=15)
learning_rate_entry.insert(0, "0.0001")

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

learnable_property_label = ttk.Label(root, text="If you are training an object or a \"something\", choose 'object'. For general style/vibe choose 'style'.")
learnable_property_combobox = ttk.Combobox(root, values=["object", "style"], width=10)
learnable_property_combobox.set("style")

num_vectors_label = ttk.Label(root, text="Number of vectors, this is a kind-of equivalent to Rank in LoRA. More vectors = more tokens are used for your embedding. (4-36 recommended)")
num_vectors_entry = ttk.Entry(root, width=10)
num_vectors_entry.insert(0, "4")

center_crop_var = tk.BooleanVar()
center_crop_checkbox = ttk.Checkbutton(root, text="Whether to center crop images before resizing to resolution.", variable=center_crop_var)

gradient_checkpointing_var = tk.BooleanVar()
gradient_checkpointing_checkbox = ttk.Checkbutton(root, text="Gradient Checkpointing (sacrifice training speed for less VRAM load, highly recommended)", variable=gradient_checkpointing_var)

enable_xformers_memory_var = tk.BooleanVar()
enable_xformers_memory_checkbox = ttk.Checkbutton(root, text="Enable xFormers (Recommended)", variable=enable_xformers_memory_var)

scale_lr_var = tk.BooleanVar()
scale_lr_checkbox = ttk.Checkbutton(root, text="Scale Learning Rate by batch size, choose if b.s. higher than 1", variable=scale_lr_var)

# Button to run the script
run_button = ttk.Button(root, text="Start Training", command=run_script)


# Checkboxes
#with_prior_preservation_checkbox.grid(row=19, column=0, sticky='w', pady=5)
center_crop_checkbox.grid(row=19, column=1, sticky='w', pady=5)
gradient_checkpointing_checkbox.grid(row=20, column=0, sticky='w', pady=5)
enable_xformers_memory_checkbox.grid(row=22, column=0, sticky='w', pady=5)
scale_lr_checkbox.grid(row=23, column=0, sticky='w', pady=5)
# Placing widgets on the window using grid layout

pretrained_model_label.grid(row=0, column=0, sticky='w', pady=5)
pretrained_model_entry.grid(row=0, column=1, pady=5)
pretrained_model_button.grid(row=0, column=2, pady=5)

train_data_dir_label.grid(row=1, column=0, sticky='w', pady=5)
train_data_dir_entry.grid(row=1, column=1, pady=5)
train_data_dir_button.grid(row=1, column=2, pady=5)

resume_from_checkpoint_label.grid(row=2, column=0, sticky='w', pady=5)
resume_from_checkpoint_entry.grid(row=2, column=1, pady=5)
resume_from_checkpoint_button.grid(row=2, column=2, pady=5)

placeholder_token_label.grid(row=3, column=0, sticky='w', pady=5)
placeholder_token_entry.grid(row=3, column=1, pady=5)

initializer_token_label.grid(row=4, column=0, sticky='w', pady=5)
initializer_token_entry.grid(row=4, column=1, pady=5)

output_dir_label.grid(row=5, column=0, sticky='w', pady=5)
output_dir_entry.grid(row=5, column=1, pady=5)
output_dir_button.grid(row=5, column=2, pady=5)

resolution_label.grid(row=6, column=0, sticky='w', pady=5)
resolution_combobox.grid(row=6, column=1, pady=5)

train_batch_size_label.grid(row=7, column=0, sticky='w', pady=5)
train_batch_size_entry.grid(row=7, column=1, pady=5)

num_train_epochs_label.grid(row=8, column=0, sticky='w', pady=5)
num_train_epochs_entry.grid(row=8, column=1, pady=5)

save_steps_label.grid(row=9, column=0, sticky='w', pady=5)
save_steps_entry.grid(row=9, column=1, pady=5)

checkpointing_steps_label.grid(row=10, column=0, sticky='w', pady=5)
checkpointing_steps_entry.grid(row=10, column=1, pady=5)

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

learnable_property_label.grid(row=18, column=0, sticky='w', pady=5)
learnable_property_combobox.grid(row=18, column=1, pady=5)

num_vectors_label.grid(row=19, column=0, sticky='w', pady=5)
num_vectors_entry.grid(row=19, column=1, pady=5)

# Checkboxes
#with_prior_preservation_checkbox.grid(row=20, column=0, sticky='w', pady=5)
center_crop_checkbox.grid(row=21, column=0, sticky='w', pady=5)
gradient_checkpointing_checkbox.grid(row=22, column=0, sticky='w', pady=5)
scale_lr_checkbox.grid(row=23, column=0, sticky='w', pady=5)
enable_xformers_memory_checkbox.grid(row=24, column=0, sticky='w', pady=5)

# Run button
run_button.grid(row=26, column=0, columnspan=3, pady=20)
# Run the main loop
root.mainloop()
