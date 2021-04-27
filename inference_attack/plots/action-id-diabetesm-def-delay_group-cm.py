
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

[y_test, y_pred] = deserialize("action-id-diabetesm-def-delay_group-cm.json")

y_pred = [y.replace('DiabetesM_', '').replace('Add', 'Add ') for y in y_pred]
y_test = [y.replace('DiabetesM_', '').replace('Add', 'Add ') for y in y_test]

classes = ['Add Proteins', 'Add Calorie', 'Add Carbs', 'Add Fat', 'Add Glucose', 'Add Insulin']

a, b, ax = plot_confusion_matrix(y_test, y_pred, labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", nolabel=False, classes=classes)
ax.tick_params(axis='both', which='major', labelsize=CONFUSION_MATRIX_XTICKS_SIZE)

write_latex_table_precision_recall_f1("action-id-diabetesm-def-delay_group-cm.tex", y_test, y_pred)
plt.tight_layout()
plt.savefig('action-id-diabetesm-def-delay_group-cm.png', format='png')
plt.savefig('action-id-diabetesm-def-delay_group-cm.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-diabetesm-def-delay_group-cm.png/eps")
