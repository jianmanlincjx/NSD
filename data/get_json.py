# import json
# import os

# def create_json_entry(text, image_file):
#     return {"text": text, "image_file": image_file}

# def generate_json_entries(image_root, text_root):
#     entries = []
#     img_list = sorted(os.listdir(image_root))

#     for img in img_list:
#         img_name = img.split('.')[0]
#         image_path = os.path.join(image_root, img)    
#         text_path = os.path.join(text_root, img_name+'.txt')
#         if os.path.exists(image_path) and os.path.exists(text_path):
#             with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 text = f.read().strip()
#             entry = create_json_entry(text, image_path)
#             entries.append(entry)
    
#     return entries

# def main(image_root1, text_root1, image_root2, text_root2, output_file):
#     entries = []

#     entries += generate_json_entries(image_root1, text_root1)
#     entries += generate_json_entries(image_root2, text_root2)

#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(entries, f, ensure_ascii=False, indent=4)
    
#     print(f"JSON file created successfully: {output_file}")

# if __name__ == "__main__":
#     image_root1 = '/data1/JM/BrushNet/data/data_train_big/image'
#     text_root1 = '/data1/JM/BrushNet/data/data_train_big/text'
#     image_root2 = '/data1/JM/BrushNet/data/data_train_small/image'
#     text_root2 = '/data1/JM/BrushNet/data/data_train_small/text'

#     main(image_root1, text_root1, image_root2, text_root2, '/data1/JM/BrushNet/data/train_combined.json')


import os
for vid in sorted(os.listdir('/data1/JM/code/BrushNet/data/data_test_anchor')):
    input_file_path = os.path.join('/data1/JM/code/BrushNet/data/data_test_anchor', vid, 'record.txt')
    output_file_path = os.path.join('/data1/JM/code/BrushNet/data/data_test_anchor', vid, 'record_new.txt')
    # 读取输入文件并处理每一行
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            modified_line = line.replace("data_test_single_mask/", "")
            output_file.write(modified_line)

    print(f"Processed file saved to {output_file_path}")