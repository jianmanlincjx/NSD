# import matplotlib.pyplot as plt

# # Sample data for the bar plot with six categories
# categories = ['Brushnet', 'CNI', 'ppt', 'Method1', 'Method2', 'Ours']
# values = [2, 5, 6, 3, 2, 10]

# # Defining different colors for each bar, with the last one being '#FF0000'
# colors = ['#FFD700', '#4682B4', '#32CD32', '#FF8C00', '#8A2BE2', '#FF0000']  # Gold, Steel Blue, Lime Green, Dark Orange, Blue Violet, Red

# # Plotting the bar plot with a taller aspect ratio
# fig, ax = plt.subplots(figsize=(5, 10))

# # Creating the bars with specified colors
# bars = ax.bar(categories, values, color=colors, edgecolor='black')

# # Customizing the x-tick labels without displaying them
# ax.set_xticks([])

# # Remove the borders except for the x-axis
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# # Hide y-axis ticks and numbers
# ax.yaxis.set_ticks_position('none')
# ax.yaxis.set_ticklabels([])

# # Saving the figure
# plt.savefig('ttts_large_font_updated.png')

# # Sample data for the second bar plot with six categories
# values = [4, 5.5, 6.5, 3, 4.5, 10]


# # Plotting the bar plot with a taller aspect ratio
# fig, ax = plt.subplots(figsize=(5, 10))

# # Creating the bars with specified colors
# bars = ax.bar(categories, values, color=colors, edgecolor='black')

# # Customizing the x-tick labels without displaying them
# ax.set_xticks([])

# # Remove the borders except for the x-axis
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# # Hide y-axis ticks and numbers
# ax.yaxis.set_ticks_position('none')
# ax.yaxis.set_ticklabels([])

# # Saving the second figure
# plt.savefig('ttts_large_font_1_updated.png')





import matplotlib.pyplot as plt
import numpy as np

# 原始数据
values1 = ['2%', '8%', '10%', '10%', '17%', '53%']  # 第一组数据
values2 = ['3%', '6%', '9%', '12%', '21%', '49%']  # 第二组数据
values3 = ['1%', '4%', '11%', '14%', '18%', '52%']  # 第三组数据

# 将百分数转换为数值形式
values1_num = [float(v.strip('%')) for v in values1]
values2_num = [float(v.strip('%')) for v in values2]
values3_num = [float(v.strip('%')) for v in values3]

# 柱子标签
labels = ['BLD', 'SDI', 'CNI', 'BrushNet', 'PP', 'Ours']

# 颜色列表
colors = ['#FFD700', '#4682B4', '#32CD32', '#FF8C00', '#8A2BE2', '#FF0000']

# 创建一个图形对象，包含3个子图，并调整其比例
fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

# 添加质感的函数
def add_bar_texture(ax, bars):
    for bar in bars:
        bar.set_edgecolor('black')  # 设置边框颜色
        bar.set_linewidth(1)  # 设置边框宽度
        bar.set_alpha(0.85)  # 设置透明度

# 第一个柱状图
bars1 = axs[0].bar(labels, values1_num, color=colors)
add_bar_texture(axs[0], bars1)
for i, v in enumerate(values1_num):
    axs[0].text(i, v + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=18, fontweight='bold')


# 第二个柱状图
bars2 = axs[1].bar(labels, values2_num, color=colors)
add_bar_texture(axs[1], bars2)
for i, v in enumerate(values2_num):
    axs[1].text(i, v + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=18, fontweight='bold')


# 第三个柱状图
bars3 = axs[2].bar(labels, values3_num, color=colors)
add_bar_texture(axs[2], bars3)
for i, v in enumerate(values3_num):
    axs[2].text(i, v + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=18, fontweight='bold')


# 自动调整布局以避免重叠
plt.tight_layout()

# 保存为图像文件
plt.savefig('textured_bar_chart.png', dpi=300)



