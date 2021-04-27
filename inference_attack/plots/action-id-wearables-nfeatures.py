
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt

[xs, ys_cla, ys_ble] = deserialize("action-id-wearables-nfeatures.json")
plt.figure()
plt.errorbar(xs, ys_cla)
plt.xlabel('Number of features')
plt.ylabel('Classifier accuracy')
#plt.title('Attacker accuracy versus Number of Features')
plt.grid()
plt.savefig('action-id-wearables-nfeatures.png', format='png')
plt.savefig('action-id-wearables-nfeatures.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-wearables-nfeatures.png/eps")
