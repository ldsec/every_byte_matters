
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.patches as patches
import matplotlib.pyplot as plt

[y_test, y_pred] = deserialize("app-id-huaweiwatch-cm-low.json")

appsByMedianTxRx = ['Battery','Reminders','DuaKhatqmAlQuran','WearCasts','DailyTracking','ASB','NoApp','HeartRate','Workout','AthkarOfPrayer','Alarm','GooglePay','Flashlight','Phone','PlayMusic','HealthyRecipes','Sleep','Medisafe','SalatTime','MapMyFitness','Calm','Citymapper','DiabetesM','Outlook','SmokingLog','Fit','Running','MapMyRun','SleepTracking','Weather','Mobills','FitBreathe','FoursquareCityGuide','FitWorkout','Glide','Translate','Qardio','Krone','FindMyPhone','KeepNotes','Shazam','Strava','Telegram','Maps','Endomondo','DCLMRadio','Lifesum','PlayStore','AppInTheAir','Bring','Spotify','Meduza','FITIVPlus','ChinaDaily','WashPost','Camera']
weirdApps = ['ASB','Alarm','AthkarOfPrayer','Battery','DailyTracking','DuaKhatqmAlQuran','Flashlight','GooglePay','HealthyRecipes','HeartRate','Medisafe','NoApp','Phone','PlayMusic','Reminders','Sleep','WearCasts','Workout']

classes=[x for x in appsByMedianTxRx if x in set(weirdApps)]

a, b, ax = plot_confusion_matrix(y_test, y_pred,nolabel=False, labelfontsize=24,  normalize=True, title="", noTextBox=True, classes=classes)
write_latex_table_precision_recall_f1("app-id-huaweiwatch-cm-low.tex", y_test, y_pred, classes=classes, labelFormat="\\app")

#ax.add_patch(patches.Rectangle((5.4, -0.4),1.2,17.9,linewidth=2,edgecolor='r',linestyle='--',facecolor='none'))
#ax.text(7, 2.5, "No App", fontsize=22, color='r')

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.tight_layout()

plt.savefig('app-id-huaweiwatch-cm-low.png', format='png')
plt.savefig('app-id-huaweiwatch-cm-low.png'.replace('.png', '.eps'), format='eps')
print("Written app-id-huaweiwatch-cm-low.png/eps")
