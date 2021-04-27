
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1, LONGRUN_LABEL_SIZE
import matplotlib.pyplot as plt

import numpy as np
import scikitplot as skplt

[linspace, rs, ps, f1s] = deserialize("longrun_p_r_f1_threshold.json")

plt.title(None)
plt.plot(linspace, ps, 'g', label="Precision")
plt.plot(linspace, rs, 'r--', label="Recall")
plt.plot(linspace, f1s, 'b:', label= "F1")

plt.xlabel("Threshold T", fontsize=LONGRUN_LABEL_SIZE)
plt.ylabel("Score", fontsize=LONGRUN_LABEL_SIZE)
plt.title("")
plt.xticks(fontsize=LONGRUN_LABEL_SIZE)
plt.yticks(fontsize=LONGRUN_LABEL_SIZE)
plt.legend(fontsize=LONGRUN_LABEL_SIZE)
plt.grid()
plt.tight_layout()
plt.savefig('longrun_p_r_f1_threshold.png', format='png')
plt.savefig('longrun_p_r_f1_threshold.png'.replace('.png', '.eps'), format='eps')
print("Written longrun_p_r_f1_threshold.png/eps")

max_ = np.argmax(f1s)
print("maximum f1 score reached at ", linspace[max_], " threshold")
print(" precision: ", ps[max_])
print(" recall: ", rs[max_])
print(" f1: ", f1s[max_])
