# K-water 현장 안전 AI 경진대회 제출 모델

OpenAI CLIP ViT-B/32 224와 Google SigLIP 2 Base/16 384의 ONNX 시각 인코더를 사용합니다. 각 인코더의 분류 확률을 같은 비율로 평균하고 임계값 0.2 이상 위험 유형 중 최대 세 개를 출력합니다. 시드는 42입니다.

```sh
python main.py --input-dir /input --output-dir /output
```

최종 예측은 /output/submission.jsonl에 기록합니다. 모델·학습 데이터 및 라이브러리의 출처와 버전은 assets.json에, 동봉 라이선스는 licenses/에 있습니다.

## 모델 및 런타임 출처

- [OpenAI CLIP](https://github.com/openai/CLIP)과 [사용한 ONNX 변환본](https://huggingface.co/Xenova/clip-vit-base-patch32). MIT. 저작권과 사용 조건은 아래 전문 및 licenses/openai-CLIP-MIT.txt에 수록했습니다.
- [Google SigLIP 2 Base/16 384](https://huggingface.co/google/siglip2-base-patch16-384)와 [사용한 ONNX 변환본](https://huggingface.co/onnx-community/siglip2-base-patch16-384-ONNX). Apache License 2.0. 전문은 licenses/siglip2-APACHE-2.0.txt에 있습니다.
- [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.26.0. MIT. licenses/onnxruntime-LICENSE 및 licenses/onnxruntime-ThirdPartyNotices.txt에 사용 조건을 수록했습니다.
- [Google FlatBuffers](https://github.com/google/flatbuffers) 25.12.19. Apache License 2.0. 전문은 licenses/flatbuffers-LICENSE.txt에 있습니다.

### OpenAI CLIP MIT License

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

