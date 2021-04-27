
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt
import matplotlib.patches as patches


[y_test, y_pred] = deserialize("app-id-transfer1-cm.json")
cm, fig, ax = plot_confusion_matrix(y_test, y_pred, nolabel=False, labelfontsize=22, normalize=True, title="", noTextBox=True, colorbar=True)
write_latex_table_precision_recall_f1("app-id-transfer1-cm.tex", y_test, y_pred)

ax.add_patch(patches.Rectangle((8.3,8.3),3.3,3.3,linewidth=2,edgecolor='r',linestyle='--',facecolor='none'))
ax.text(10.5, 7.5, "OS Fit packages", fontsize=22, color='r')

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig('app-id-transfer1-cm.png', format='png')
plt.savefig('app-id-transfer1-cm.png'.replace('.png', '.eps'), format='eps')
print("Written app-id-transfer1-cm.png/eps")
