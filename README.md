*Made using scripts from https://github.com/huggingface/diffusers*

**NOTE: The LoRAs only work for ComfyUI at this time, A1111 and other
UIs load LoRAs differently.**

**Python installation required, I recommend Python 3.10.6 but slightly
newer/older versions should work as well.**

This is a small training GUI for Stable Diffusion 1.5 and XL
It is intended to simplify the Stable Diffusion training process,
and thus make it more accessible for a wider audience. Currently this
GUI is for Nvidia GPUs on Windows, you *may* be able to get it to
work on AMD GPUs using the ROCm version of PyTorch on Linux by editing
the .bat files and turning them into .sh files.

Nonetheless, there are some concepts you should know in order to
use this GUI effectively.

It can train SD1.5 LoRAs using as little as 4GB VRAM (6GB VRAM GPU recommended)

It can train SDXL LoRAs using as little as 12GB VRAM (16GB VRAM GPU recommended)

I am going to try to keep this as short as possible, if you want
more "in depth" information, there are many tutorials and guides
online that explain Stable Diffusion LoRA/Dreambooth training in
greater detail. Many of the concepts in those guides apply to this
GUI as well.

*HOW TO START*

### INSTALLATION
1. Run installer.bat
2. Run sd15_fp16_model_download.bat AND/OR sdxl_model_download.bat
   to conveniently download the diffusers base models
3. Prepare your dataset of training images
4. Put them in the right folder/s
5. Run the .bat file of the trainer of your choice

### TRAINING IMAGES
Start building a collection of training images. Here are some
criteria:

-They should be all the same format (.png recommended) 

-For SD1.5, at least one dimension should always be 512px 
    (512x640, 384x512, 256x512, etc.)
    
-If you want to keep it simple, crop all images to 512x512

-Note that this makes it more difficult to generate good
    images in other aspect ratios later on, but it is good to
    start off simple.
    
-For SDXL the standard resolution is 1024x1024

-Avoid pixelated, blurry, and "chopped up" images

-The images should be diverse and unique, but generally have
    the same style (don't put a photo of a dog into a bunch
    of anime images)
    
-Tip: select all images, then rename one of them to a random letter,
    this is a fast way to number them, although it is not required.
    
Once you are done building your image dataset, throw them into the
"instance-imgs" folder for SD1.5 or the "instance-imgs-sdxl" for SDXL.

### INSTANCE PROMPT AND CLASS PROMPT
This is what you are going to add to the prompt later on, the instance
prompt should be some random unique word that is not an existing token.
For example, "fsk49", "mycat9", "thbanana" are all examples of instance
prompts. The class prompt should be a one-word token that roughly 
describes your training images. For example, if you train images of
cats, the class prompt should be "cat", if your images are anime girls,
your class prompt should be "1girl", because it is a Danbooru tag that
describes a single anime girl. The rules are not so strict however,
you can even enter "1girl" as a class prompt but still choose to use
"itok30 anime girl" instead of "itok30 1girl", even if you trained on
the latter.

Remember to not forget your class token when generating images!

### BASE MODEL PATH
I recommend you keep the standard sd1.5/sdxl base model, even if you train
on something like anime images. Just use the LoRA with an anime model
later on.

### BATCH SIZE
How many images to train per step. When you have a lot of VRAM left, 
it may seem like a good idea to train 2, 4, or even more images per step,
because this would slash down the total training time to 1/2, 1/4, or
even less, but unfortunately it can take a hit on the quality of the
LoRA that comes out. I recommend leaving it at 1 for starters, but you
can try out 2 or 4 later on. Above 4 the model quality will significantly
decrease. Remember to tick "Scale Learning Rate" when doing this.

### OUTPUT LORA
You don't have to keep your LoRA called "pytorch_lora_weights", you are
free to rename it to whatever you want.

### EPOCHS AND SAVING CHECKPOINTS
**(It is assumed that you have kept the Batch Size at 1)**

To keep it simple, one Epoch is one "revolution" of your training images,
so for 24 instance images one Epoch is 24 steps. Two Epochs are 48 steps.
You can calculate the total amount of steps using this formula:

**Total Steps = Number of Epochs * (Number of Training Images)**

It is hard to say how many Epochs are sufficient, because it depends on
how many images you have for training. For 10 images, it is recommended that you train for
32-256 Epochs. For 100 images, maybe 8-64 is sufficient. This is why
it makes sense to set a high Epoch number, calculate the total
amount of steps using the above formula, and then enter into
"Save checkpoint every n steps" some even number fraction of your total
steps (if your total steps are 6400, you can enter 1600 or 800)

So even if your final LoRA is complete garbage and overfitted, you can
still try out the LoRAs that were saved in between.

For Dreambooth model training with class images you need to include the
number of class images with the total step calculation.

### Use 8-bit-AdamW
This optimizer saves a lot of VRAM over AdamW, at only a minimal loss in
quality. However, it is reported to not work for SDXL and may only
work for 30XX/400XX series Nvidia GPUs.



### WHAT IS A LoRA?
It is basically a smaller model that is "mounted" on the main
Stable Diffusion checkpoint. It is practical because it takes a
lot less VRAM to train than a full Dreambooth



### Dreambooth technique vs. Fine-tune technique
This implementation only supports the Dreambooth technique. The
advantage is that you don't need any .txt captions along with
your images, the disadvantage is that you can't "fine-tune". I
am not an expert in this subject, but I will explain it like
this: Imagine you train a LoRA on a bunch of anime images. The
Dreambooth technique will learn the "style" of your images, 
so the images you end up generating will have the same anime
"style" as your training images.

The fine-tune technique is more complicated. It is based on
captions and allows a LoRA to more closely approximate your
training images, it also greately increases prompt adherence.
As an example the "Waifu-Diffusion" model was trained on
Danbooru captions and keywords, which is why Danbooru tags
work a lot better with a fine-tune like "Waifu-Diffusion".

If you are interested in fine-tuning, check out bmaltais's
koyha_ss fork: 

https://github.com/bmaltais/kohya_ss

It is more complicated, but comes with a huge amount of features.
