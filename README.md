# osc-vad
基于onnxruntime的高性能语音端点检测。


## 特点
- [x] 内置模型，无需下载。
- [x] 自动选择cpu或者gpu。


## 模型

- [x] FSMN（16K采样率）
- [ ] Silero

## 安装
```shell
pip install osc-vad
```

## 使用
```python
from osc_vad import FSMN
vad = FSMN()
for chunk in chunks:
    segments = vad.process_chunk(chunk)
    for segment in segments:
        print(segment)
```

## 性能

| 模型 | 采样率 | RTF | Speedup Rate | device |
| --- | --- | --- | --- | --- |
| FSMN | 16K | 0.0038 | 263.64 | macbook pro m4 |