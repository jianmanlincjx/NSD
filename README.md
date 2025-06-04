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
```


### Data Download
NSD uses [BrushData and BrushBench](https://github.com/TencentARC/BrushNet?tab=readme-ov-file) for training and testing. You can download the dataset through this link. At the same time, NSD proposes an indoor dataset for specialized self-styled editing of indoor scenes. This dataset is still being organized. Once the dataset is ready, you can organize it in JSON format within the "data" folder for training.


## Running Scripts

### Training 🤖
You can train with segmentation mask using the script:

```bash
# sd v1.5
# # sd v1.5 follow IP-adapter
accelerate launch --num_processes 3 examples/NSD/train_NSD.py \
--pretrained_model_name_or_path /data1/JM/code/NSD/pretrain_model/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_segmentationmask \
--resolution 512 \
--learning_rate 1e-5 \  
--train_batch_size 2 \
--tracker_project_name NSD \
--report_to tensorboard \
--resume_from_checkpoint latest \
--checkpointing_steps 100000  \
--json_file /data1/JM/code/NSD/data/train_small.json \
--brushnet_model_name_or_path /data1/JM/code/NSD/pretrain_model/segmentation_mask_brushnet_ckpt \
--mixed_precision 'fp16' \
--validation_image /data1/JM/code/NSD/data/data_train_small/image/000075.png \
--validation_mask /data1/JM/code/NSD/data/data_train_small/mask/chair/000075.png \
--validation_prompt 'A delicate sofa in the room. ' \
--validation_steps 1000 \
--image_encoder_path /data1/JM/code/NSD/pretrain_model/image_encoder
```
### Inference 🎨
```bash
python examples/NSD/test_NSD.py
```
### 🚨 Important Notes

- Ensure all pre-trained models are correctly downloaded and placed in the specified locations
- Training can be performed on a GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090)
- For inference, a GPU with 12GB VRAM is sufficient
- CUDA 11.7 or higher is recommended

