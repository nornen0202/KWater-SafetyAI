# FloodNet 공개 특징 306장

`pilot-features.npz`는 FloodNet 공개 훈련 사진 192장과 시험 사진 114장에서 추출한 512차원 CLIP 특징, 침수 이진 표시, 원래 분할, 선정 집계다. 특징을 만든 시각 모델은 Xenova의 양자화된 CLIP ViT-B/32이며 리비전은 `6b7c523703ae26e39be4b03774966f8571c7f523`이다. 파일 SHA-256은 `50ec39f21c94e861de24a90b4289e5fed48d24ab86753f0e63610475c88ee3cc`이다.

FloodNet 연구진인 Maryam Rahnemoonfar 외가 허리케인 하비 이후 촬영한 미국 항공 영상을 제공했다. 원 자료는 Community Data License Agreement – Permissive 1.0으로 배포됐다. 논문은 *FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding*, IEEE Access, 2021이다. [배포처](https://github.com/BinaLab/FloodNet-Supervised_v1.0) · [이용 조건 원문](https://cdla.io/permissive-1-0/) · [논문](https://doi.org/10.1109/ACCESS.2021.3090981) · [시각 모델](https://huggingface.co/Xenova/clip-vit-base-patch32)

[실행 노트북](../../tools/floodnet_cloud_pilot.ipynb)은 이 파일 하나로 중심점과 선형 모델을 재현한다. 특징의 원본 이미지 선정 및 마스크 기준은 [사전 실험 설계](../../docs/external-pilot-protocol.md)에 있다.
