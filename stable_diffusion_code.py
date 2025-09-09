from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import os
# Load the pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# Load the base image and mask image
# Replace 'base_image.png' and 'mask_image.png' with your actual file paths
image_folder_path = '/home/work/.sina/Ptrain_input' #"TRAIN_00003.png"
mask_folder_path = '/home/work/.sina/Ptrain_mask' #"TRAIN_00003_mask.png"

image_list = os.listdir(image_folder_path)
mask_list = os.listdir(mask_folder_path)

for i in range(len(image_list)):

    image = Image.open(os.path.join(image_folder_path, image_list[i])).convert("RGB")
    #print(mask_list[i])
    mask_image = Image.open(os.path.join(mask_folder_path, mask_list[i])).convert("L")
    prompt_1 = "Fill vacant space"
    prompt_2 = "Please fill the vacant space by considering the pixels around the mask."
    prompt_3 = "Please fill the vacant space by considering the pixels around the mask. Please exclude any text or letters in vacant space."
    output_image_1 = pipe(prompt=prompt_1, image=image, mask_image=mask_image).images[0]
    output_image_2 = pipe(prompt=prompt_2, image=image, mask_image=mask_image).images[0]
    output_image_3 = pipe(prompt=prompt_3, image=image, mask_image=mask_image).images[0]

    os.makedirs('./train_output_1', exist_ok = True)
    os.makedirs('./train_output_2', exist_ok = True)
    os.makedirs('./train_output_3', exist_ok = True)

    output_image_1.save(os.path.join('./train_output_1', image_list[i]))
    output_image_2.save(os.path.join('./train_output_2', image_list[i]))
    output_image_3.save(os.path.join('./train_output_3', image_list[i]))

"""
# Load the images using PIL
image = Image.open(image_path).convert("RGB")
mask_image = Image.open(mask_path).convert("L")  # Ensure the mask is in grayscale

# Ensure the mask follows the convention: white (255) for inpainting and black (0) for keeping as is
# If you need to manually create a mask, you can use the code below:
# from PIL import ImageDraw
# mask_image = Image.new("L", image.size, 0)  # Start with an all-black mask
# draw = ImageDraw.Draw(mask_image)
# draw.rectangle((50, 50, 150, 150), fill=255)  # Draw a white rectangle for inpainting

# Define the prompt
#prompt = "Fill vacant space or blank area"



# Perform inpainting
output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

# Save the output image
output_image.save("./TRAIN_00003_output.png")
print("Output image saved to './TRAIN_00003_output.png'")
"""