
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

[y_test, y_pred] = deserialize("action-id-wearables-def-pad-cm.json")

def replace(s):
    s = s.replace('SamsungGalaxyWatch', 'GalaxyWatch')
    return s

y_test = [replace(y) for y in y_test]
y_pred = [replace(y) for y in y_pred]

cm, fig, ax = plot_confusion_matrix(y_test, y_pred, nolabel=False,labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", noTextBox=True, figsize=(9, 8), colorbar=True)
    
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.tight_layout()


write_latex_table_precision_recall_f1("action-id-wearables-def-pad-cm.tex", y_test, y_pred)
plt.tight_layout()
plt.savefig('action-id-wearables-def-pad-cm.png', format='png')
plt.savefig('action-id-wearables-def-pad-cm.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-wearables-def-pad-cm.png/eps")
