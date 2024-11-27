#from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="hpu")
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
device = args.device

links = ["https://raw.githubusercontent.com/opea-project/GenAIExamples/refs/heads/main/VisualQnA/ui/svelte/static/favicon.png","https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg","https://raw.githubusercontent.com/Ikaros-521/digital_human_video_player/refs/heads/main/static/imgs/1.png","https://raw.githubusercontent.com/Ikaros-521/digital_human_video_player/refs/heads/main/static/imgs/2.png","https://raw.githubusercontent.com/Spycsh/assets/refs/heads/main/OPEA%20Telemetry.jpg"]
links.reverse() # The last link output exceed max_tokens default 128, let it warmup first
#|----input---|---pad by processor ----------------------|--decoding max_new_tokens---------------|
#|---------------------------4096------------------------|----------------------128---------------|


if device == "hpu":
    if not args.baseline:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()
    else: # naive implementation
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.gpu_migration
# default: Load the model on the available device(s)
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto",
).to(device)

print(device)
# Do not use wrap_in_hpu_graph since the repeated inference will cause an error on HPU
#if device == "hpu":
#    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
#    model = wrap_in_hpu_graph(model)
#    print(f"Use static generation {not args.baseline}")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
for link in links:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": link,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    '''
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    '''
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,return_tensors="pt",padding='max_length',max_length=4096)
    print(inputs['input_ids'].shape)
    inputs = inputs.to(device)
    # Inference: Generation of the output
    for i in range(1):
        start = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=128,)
        print(time.time() - start)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
