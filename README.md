# Neural Scene Designer: Self-Styled Semantic Image Manipulation

Detailed inference and training procedures coming soon.

![PDF 图像](https://github.com/jianmanlincjx/NSD/blob/main/NSD_result.png)

## Getting Started

### Environment Requirement 🌍
NSD has been implemented and tested on Pytorch 1.12.1 with Python 3.9.

#### Clone the repo and setup environment
```bash
git clone https://github.com/jianmanlincjx/NSD.git
conda create -n diffusers python=3.9 -y
conda activate diffusers
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://
pip install -e .
cd examples/NSD/
pip install -r requirements.txt


### Data Download
NSD uses BrushData and BrushBench for training and testing. You can download the dataset through this link [https://github.com/TencentARC/BrushNet?tab=readme-ov-file]. At the same time, NSD proposes an indoor dataset xxx for specialized self-styled editing of indoor scenes. This dataset is still being organized. Once the dataset is ready, you can organize it in JSON format within the "data" folder for training.
