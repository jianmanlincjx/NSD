# import pandas as pd


### 获取并集
# # List of CSV file paths
# csv_files = [
#     '/data1/JM/code/BrushNet/data/baseline_BLD/evaluation_result.csv',
#     '/data1/JM/code/BrushNet/data/baseline_SD/evaluation_result.csv',
#     '/data1/JM/code/BrushNet/data/baseline_brushnet/evaluation_result.csv',
#     '/data1/JM/code/BrushNet/data/baseline_controlnet/evaluation_result.csv',
#     '/data1/JM/code/BrushNet/data/baseline_ours/evaluation_result.csv',
#     '/data1/JM/code/BrushNet/data/baseline_ppt/evaluation_result.csv'
# ]

# # Function to extract base_name from image_path
# def extract_base_name(image_path):
#     return '/'.join(image_path.split('/')[-3:])

# # Step 1: Extract base_name lists from each CSV
# base_name_lists = []
# dfs = []
# for file in csv_files:
#     df = pd.read_csv(file)
#     df['base_name'] = df['image_path'].apply(extract_base_name)
#     base_name_lists.append(set(df['base_name']))
#     dfs.append(df)

# # Step 2: Find the common base_names across all CSV files
# common_base_names = set.intersection(*base_name_lists)

# # Step 3: Filter each CSV to keep only rows with common base_names and print deleted rows
# for df, file in zip(dfs, csv_files):
#     filtered_df = df[df['base_name'].isin(common_base_names)]
#     deleted_rows = df[~df['base_name'].isin(common_base_names)]
    
#     if not deleted_rows.empty:
#         print(f"Deleted rows from {file}:")
#         print(deleted_rows)
    
#     filtered_df.drop(columns=['base_name'], inplace=True)
#     filtered_df.to_csv(file, index=False)

# print("Filtered CSV files have been saved.")


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################



import pandas as pd

# Define the file paths
file_paths = [
    '/data1/JM/code/BrushNet/data/baseline_BLD/evaluation_result.csv',
    '/data1/JM/code/BrushNet/data/baseline_SD/evaluation_result.csv',
    '/data1/JM/code/BrushNet/data/baseline_brushnet/evaluation_result.csv',
    '/data1/JM/code/BrushNet/data/baseline_controlnet/evaluation_result.csv',
    '/data1/JM/code/BrushNet/data/baseline_ours/evaluation_result.csv',
    '/data1/JM/code/BrushNet/data/baseline_ppt/evaluation_result.csv'
]


# Read the data from the files
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Check and print the columns of each DataFrame
for idx, df in enumerate(dataframes):
    print(f"Columns in file {file_paths[idx]}: {df.columns.tolist()}")

# Ensure all files have the same structure
columns = dataframes[0].columns

for df in dataframes:
    assert df.columns.equals(columns), "The files do not have the same columns."

# Define the comparison rule
def comparison_score(row):
    return (
        row['Image Reward'] + 
        row['HPS V2.1'] + 
        row['Aesthetic Score'] + 
        row['PSNR'] - 
        row['LPIPS'] - 
        row['MSE'] + 
        row['CLIP Similarity'] + 
        row['style_con']
    )

# Calculate the comparison score for each row in each dataframe
for df in dataframes:
    df['Comparison Score'] = df.apply(comparison_score, axis=1)

# Extract the relevant dataframe for 'baseline_ours'
baseline_ours_df = dataframes[4]

# Compare each row of 'baseline_ours' with corresponding rows in other dataframes and rank
rankings = []
for index, row in baseline_ours_df.iterrows():
    scores = [df.loc[index, 'Comparison Score'] for df in dataframes]
    baseline_ours_score = scores[4]
    rank = sorted(scores, reverse=True).index(baseline_ours_score) + 1
    if rank <= 3:
        rankings.append(row['image_path'])

# Convert rankings to a DataFrame and save to a CSV if needed
top_ranked_df = pd.DataFrame(rankings, columns=["Top Ranked Image Paths"])
top_ranked_df.to_csv('top_ranked_image_paths.csv', index=False)

# Display the top ranked image paths
print(top_ranked_df)