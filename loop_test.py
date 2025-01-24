# loop test to check whether HPU graph is allocated correctly
# to prevent FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size ...


from transformers import AutoProcessor, pipeline
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()
from qwen_vl_utils import process_vision_info
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch
device = "hpu"
model = "Qwen/Qwen2-VL-7B-Instruct"
torch_dtype = torch.bfloat16
processor = AutoProcessor.from_pretrained(model)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained(model, torch_dtype="auto").to(device)

model = wrap_in_hpu_graph(model)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Write a joke for the image."},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding='max_length',
    max_length=4096,
    return_tensors="pt",
)
inputs = inputs.to(device)

for i in range(5):
    generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True, cache_implementation="static", static_shapes=True, use_flash_attention=True)


    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
