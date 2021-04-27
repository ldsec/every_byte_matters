
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt

from lib.defenses import CostAggregate
from lib.plot_builder import write_latex_table_defense
import numpy as np

defense_results2 = deserialize("action-id-wearables-def.json")
defense_results = dict()

for defense in defense_results2:
    defense_results[defense] = dict(scores=defense_results2[defense]['scores'],
    costs=CostAggregate().from_serialized_array(defense_results2[defense]['costs']))


write_latex_table_defense("action-id-wearables-def.tex", defense_results, labelFormat="\\defense", headerRow=False)

print("Written device-id-defenses-ble.tex")
sys.exit(0)