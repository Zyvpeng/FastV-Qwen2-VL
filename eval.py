from datasets import load_dataset, Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained("./qwen-vl2-2b", torch_dtype=torch.float16,
                                                        use_flash_attention_2=False, resume_download=True).to('cuda')
processor = AutoProcessor.from_pretrained("./qwen-vl2-2b", torch_dtype=torch.float16)
# data = load_dataset("merve/vqav2-small")['validation']
# data = data.select(range(1000))
# data = Dataset.load_from_disk('/root/autodl-tmp/vlm_compress/icae/code/icae_v2/utils/gqa_testdev_1000').select(
#     range(0, 1000))
data = load_dataset("lmms-lab/MME")['test']
category = {}
res = {}
for _ in data:
    if _['category'] in category:
        category[_['category']] += 1
    else:
        category[_['category']] = 1
        res[_['category']] = []

print(category)
print(data)
# 280,280-> 721
# 224,224-> 695
# icae -> 731

# icae ->737
# 560 -> 800
count = 0
res_list = []
for d in data:
    # print(d)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": d["image"],
                    "resized_height": 560,
                    "resized_width": 560,
                },
                {"type": "text", "text": d["question"]}
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_input, video_input = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_input,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to('cuda')
    import time

    start = time.process_time()
    # for i in range(128):
    #     model(**inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    end = time.process_time()
    print(f"decode时间:{end - start:.6f}s")
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"out:{output_text[0]}")
    output_text[0] = output_text[0].lower()
    d['answer'] = d['answer'].lower()
    print(f"gt:{d['answer']}")
    if output_text[0] == d['answer'] or output_text[0] in d['answer'] or d['answer'] in output_text[0]:
        res[d['category']].append(True)
        res_list.append(True)
        count += 1
    else:
        res_list.append(False)
        res[d['category']].append(False)

# acc
print(f"acc:{sum(res_list)}:{len(res_list)}")

# acc+
cc = 0
for i in range(0, len(res_list), 2):
    if res_list[i] == True and res_list[i + 1] == True:
        cc += 1

print(f"acc++:{cc}:{len(res_list)}")

score = 0
for k, v in category.items():
    acc = sum(res[k])
    acc_plus = 0
    for i in range(0, v, 2):
        if res[k][i] == True and res[k][i + 1] == True:
            acc_plus += 1
    print(f"{k}: acc{100 * acc / v}   acc+{100 * acc_plus / (v / 2)}")
    score += 100 * acc / v + 100 * acc_plus / (v / 2)

print(score)
