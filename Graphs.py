import matplotlib.pyplot as plt
import numpy as np

labels = ['VGG13BiConvLSTM', 'VGG13', 'FightNetConv3D', 'ThreeStreamsLSTM', '3DCNN', 'AlexNetConvLSTM', 'HoughForestCNN',
          'FlowGatedNetwork', 'FineTuningMobile', 'XceptionAttention10', 'XceptionAttention5', 'SELayerC3D', 'ProposedMethod']

values_hf = [96.54, 96.96, 97.0, 93.9, 96.0, 97.1, 94.6, 48.1, 87.0, 97.5, 98.0, 99.0, 99.0]
values_mf = [100.0, 100.0, 100.0, 0.0, 99.9, 100.0, 99.0, 59.0, 99.5, 100.0, 100.0, 0.0, 100.0]
values_vf = [92.18, 90.63, 0.0, 0.0, 98.0, 94.57, 0.0, 50.0, 0.0, 0.0, 0.0, 98.08, 94.0]

values_hf_error = [1.01, 1.08, 0.0, 0.0, 0.0, 0.55, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
values_mf_error = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
values_vf_error = [3.29, 2.82, 0.0, 0.0, 0.0, 2.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]

x = np.arange(len(labels))
width = 0.2

fig = plt.figure(figsize=(23, 10))
ax = fig.add_subplot()

rects1 = ax.bar(x - width, values_hf, width, label='HockeyFights Dataset', color='r', yerr=values_hf_error)
rects2 = ax.bar(x, values_mf, width, label='MoviesFights Dataset', color='g', yerr=values_mf_error)
rects3 = ax.bar(x + width, values_vf, width, label='ViolentFlows Dataset', color='b', yerr=values_vf_error)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores', fontsize=16)
ax.set_title('Test Scores by dataset', fontsize=16)
plt.yticks(fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=16)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width(), height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()
