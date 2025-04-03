# Neural Scene Designer: Self-Styled Semantic Image Manipulation

Detailed inference and training procedures coming soon.

![PDF 图像](https://github.com/jianmanlincjx/NSD/blob/main/NSD_result.png)

## Getting Started

### Environment Requirement 🌍
BrushNet has been implemented and tested on Pytorch 1.12.1 with Python 3.9.

#### Clone the repo:
```bash
git clone https://github.com/TencentARC/BrushNet.git
conda create -n diffusers python=3.9 -y
conda activate diffusers
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://
pip install -e .
cd examples/brushnet/
pip install -r requirements.txt
