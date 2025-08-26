from transformers import pipeline

# Load the question-answering pipeline
#qa_pipeline = pipeline("question-answering",
#model="deepset/roberta-base-squad2")

# Correct input format
#result = qa_pipeline(question="What is Hugging Face?",
#context="Hugging Face is a platform for NLP.")
#print(result)

import torch
from diffusers import DiffusionPipeline

# Load the diffusion pipeline from a pretrained model, using bfloat16 for tensor types.
pipe = DiffusionPipeline.from_pretrained(
    "shuttleai/shuttle-3-diffusion", torch_dtype=torch.bfloat16
).to("cuda")

# Uncomment the following line to save VRAM by offloading the model to CPU if needed.
# pipe.enable_model_cpu_offload()

# Uncomment the lines below to enable torch.compile for potential performance boosts on compatible GPUs.
# Note that this can increase loading times considerably.
# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(
#     pipe.transformer, mode="max-autotune", fullgraph=True
# )

# Set your prompt for image generation.
prompt = "A cat holding a sign that says hello world"

# Generate the image using the diffusion pipeline.
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=4,
    max_sequence_length=256,
    # Uncomment the line below to use a manual seed for reproducible results.
    # generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Save the generated image.
image.save("shuttle.png")