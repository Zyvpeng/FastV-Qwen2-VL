from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load the model (on the available device(s))
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./qwen-vl2-2b", torch_dtype="auto", device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained("./qwen-vl2-2b", torch_dtype=torch.float16)

# Prepare the image input
image_path = "E:/VLM-Compress/HFT-learn/demo.jpeg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
                "resized_height": 560,
                "resized_width": 560,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Prepare for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
# print(text)
# text = "<|im_start|><|image_pad|><|im_end|>"
# text = "<|im_end|>\n<|im_start|>assistant\n"
# Prepare inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to('cuda')
# inputs = inputs.to("cuda")

# Inference: Generate output (image description or other tasks)
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

