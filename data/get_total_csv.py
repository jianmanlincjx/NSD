import pandas as pd
import os

# ['baseline_BLD', 'baseline_SD', 'baseline_brushnet', 'baseline_controlnet', 'baseline_ours', 'baseline_ppt']
for name in ['baseline_BLD', 'baseline_SD', 'baseline_brushnet', 'baseline_controlnet', 'baseline_ours', 'baseline_ppt']:

    data_name = name

    root = '/data1/JM/code/BrushNet/data/Baseon_4K_dataset'
    # os.system(f'rm -rf {root}/{data_name}/evaluation_result.csv')
    # os.system(f'rm -rf {root}/{data_name}/evaluation_result_mean.csv')
    vid_list = sorted(os.listdir(f'{root}/{data_name}'))
    vid_list = [i for i in vid_list if not i.endswith('.csv')]
    folder_paths = [os.path.join(root, data_name, vid, 'evaluation_result_SS.csv') for vid in vid_list]
    combined_df = pd.DataFrame()

    for file_path in folder_paths:
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df.to_csv(f'{root}/{data_name}/evaluation_result_SS.csv', index=False)
    filtered_df = combined_df.iloc[:, 2:-1]
    mean_df = filtered_df.mean().to_frame(name='mean') 
    mean_df.to_csv(f'{root}/{data_name}/evaluation_result_mean_SS.csv')
    print("CSV files have been combined and mean calculated successfully.")


# data_name = 'baseline_SD'
# root = '/data1/JM/code/BrushNet/data'
# vid_list = sorted(os.listdir(f'{root}/{data_name}'))
# folder_paths = [os.path.join(root, data_name, vid, 'evaluation_result.csv') for vid in vid_list]
# combined_df = pd.DataFrame()
# for file_path in folder_paths:
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
#         combined_df = pd.concat([combined_df, df], ignore_index=True)
# combined_df.to_csv(f'{root}/{data_name}/evaluation_result.csv', index=False)
# filtered_df = combined_df.iloc[:, 2:]
# mean_df = filtered_df.mean().to_frame(name='mean') 
# mean_df.to_csv(f'{root}/{data_name}/evaluation_result_mean.csv')
# print("CSV files have been combined and mean calculated successfully.")