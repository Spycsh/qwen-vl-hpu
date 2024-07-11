# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.generation import GenerationConfig
import torch
import time
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="hpu", trust_remote_code=True).eval()
# use cuda device
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

"""Long Warmup"""
query = tokenizer.from_list_format([
    {'image': 'https://raw.githubusercontent.com/opea-project/GenAIExamples/main/VisualQnA/ui/gradio/resources/waterview.jpg'},
    {'text': '图里有些什么？'},
])

for i in range(1):
    s=time.time()
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    print(time.time() - s)


# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么？'},
])

# 测试 无history性能
for i in range(1):
    s=time.time()
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    print(time.time() - s)

# 测试 有history性能
for i in range(2):
    s=time.time()
    response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
    print(response)
    print(history)
    print(time.time() - s)

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
 image.save('1.jpg')
else:
 print("no box")

query = tokenizer.from_list_format([
    {'image': 'https://raw.githubusercontent.com/opea-project/GenAIExamples/main/VisualQnA/ui/gradio/resources/extreme_ironing.jpg'},
    {'text': '这是什么？'},
])

for i in range(1):
    s=time.time()
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    print(time.time() - s)

query = tokenizer.from_list_format([
    {'image': 'https://www.iaea.org/sites/default/files/styles/third_page_width_portrait_2_3/public/imaging-chinese-1.jpg?itok=xn6xAzo2'},
    {'text': '这是什么？'},
])

for i in range(1):
    s=time.time()
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    print(time.time() - s)
