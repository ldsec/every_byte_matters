
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

[y_test, y_pred] = deserialize("chipset-id-cm.json")

a, b, ax = plot_confusion_matrix(y_test, y_pred, labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", nolabel=False)
write_latex_table_precision_recall_f1("chipset-id-cm.tex", y_test, y_pred)
ax.tick_params(axis='both', which='major', labelsize=CONFUSION_MATRIX_XTICKS_SIZE)

plt.tight_layout()

plt.savefig('chipset-id-cm.png', format='png')
plt.savefig('chipset-id-cm.png'.replace('.png', '.eps'), format='eps')
print("Written chipset-id-cm.png/eps")
