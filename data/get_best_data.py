import os

vid_list = ['baseline_BLD', 'baseline_SD', 'baseline_brushnet', 'baseline_controlnet', 'baseline_ours', 'baseline_ppt']
sub_vid_list = sorted(os.listdir('/data1/JM/code/BrushNet/data/Baseon_4K_dataset/baseline_BLD'))

anchor_root = '/data1/JM/code/BrushNet/data/Baseon_4K_dataset/data_test_new'

for vid in vid_list:
    for sub_vid in sub_vid_list:
        save_text_path = os.path.join('/data1/JM/code/BrushNet/data/Baseon_4K_dataset', vid, sub_vid, 'record_new.txt')
        source_text_path = os.path.join(anchor_root, sub_vid, 'record_new.txt')
        os.system(f'cp {source_text_path} {save_text_path}')