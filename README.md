BKC on HPU

# Qwen-VL 1
```
docker run -itd -p 8091:80  --runtime=habana -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host vault.habana.ai/gaudi-docker/1.16.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest

pip install tiktoken matplotlib
pip install einops transformers_stream_generator
pip install accelerate
pip install --upgrade-strategy eager optimum[habana]

git clone https://github.com/Spycsh/optimum-habana.git
cd optimum-habana
git checkout enable_qwen_vl
export PYTHONPATH=/root:/usr/lib/habanalabs/:/optimum-habana/

python test_baseline_cpu.py
cp ./modeling_qwen.py /root/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-VL-Chat/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/modeling_qwen.py

python test_hpu.py
```

Rough perf comparison (1 Gaudi card vs. 8380 xeon cpu)

hpu: 80ms/token

cpu: 650ms/token


# Qwen-VL 2

Use [my branch](https://github.com/Spycsh/optimum-habana/tree/qwen2_vl), set PYTHONPATH correctly

```
python qwen2_vl.py
```

Make sure you fix the `max_new_tokens`. After the first warmup, with `max_new_tokens` 64/128, the last four latencies should be about 5 to 10s. close to A100. 4 to 7 times faster than cpu.

> Take care if there is a perf gap, please make sure `max_length` and `max_new_tokens` are set reasonably.
