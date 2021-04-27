
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

weirdApps = ['ASB','Alarm','AthkarOfPrayer','Battery','DailyTracking','DuaKhatqmAlQuran','Flashlight','GooglePay','HealthyRecipes','HeartRate','Medisafe','NoApp','Phone','PlayMusic','Reminders','Sleep','WearCasts','Workout']
nonWeirdApps = ['AppInTheAir','Bring','Calm','Camera','ChinaDaily','Citymapper','DCLMRadio','DiabetesM','Endomondo','FITIVPlus','FindMyPhone','Fit','FitBreathe','FitWorkout','FoursquareCityGuide','Glide','KeepNotes','Krone','Lifesum','MapMyRun','Maps','Meduza','Mobills','Outlook','PlayStore','Qardio','Running','SalatTime','Shazam','SleepTracking','SmokingLog','Spotify','Strava','Telegram','Translate','MapMyFitness','WashPost','Weather']
twoClasses = weirdApps
twoClasses.extend(nonWeirdApps)

appsByMedianTxRx = ['Battery','Reminders','DuaKhatqmAlQuran','WearCasts','DailyTracking','ASB','NoApp','HeartRate','Workout','AthkarOfPrayer','Alarm','GooglePay','Flashlight','Phone','PlayMusic','HealthyRecipes','Sleep','Medisafe','SalatTime','MapMyFitness','Calm','Citymapper','DiabetesM','Outlook','SmokingLog','Fit','Running','MapMyRun','SleepTracking','Weather','Mobills','FitBreathe','FoursquareCityGuide','FitWorkout','Glide','Translate','Qardio','Krone','FindMyPhone','KeepNotes','Shazam','Strava','Telegram','Maps','Endomondo','DCLMRadio','Lifesum','PlayStore','AppInTheAir','Bring','Spotify','Meduza','FITIVPlus','ChinaDaily','WashPost','Camera']
appsByMeanTxRx = ['Battery','DailyTracking','Phone','ASB','GooglePay','HealthyRecipes','SalatTime','Workout','MapMyFitness','AthkarOfPrayer','Flashlight','Fit','Calm','FitBreathe','Medisafe','Running','NoApp','DiabetesM','Alarm','Mobills','SleepTracking','Outlook','Reminders','FoursquareCityGuide','HeartRate','Citymapper','Weather','Sleep','MapMyRun','Glide','DuaKhatqmAlQuran','PlayMusic','Translate','Qardio','KeepNotes','Krone','Strava','Telegram','FindMyPhone','Maps','SmokingLog','Shazam','Endomondo','Lifesum','DCLMRadio','FitWorkout','AppInTheAir','Bring','PlayStore','Spotify','Meduza','ChinaDaily','FITIVPlus','WearCasts','WashPost','Camera']
appsByStdTxRx = ['WearCasts','FitWorkout','Camera','SmokingLog','DuaKhatqmAlQuran','Shazam','PlayStore','Reminders','HeartRate','WashPost','Endomondo','DCLMRadio','PlayMusic','AppInTheAir','Sleep','FindMyPhone','Meduza','FITIVPlus','Krone','MapMyRun','Weather','NoApp','Alarm','Medisafe','Spotify','Lifesum','Outlook','Strava','FoursquareCityGuide','Maps','Qardio','Bring','Telegram','Translate','Glide','Workout','AthkarOfPrayer','SleepTracking','KeepNotes','Calm','Citymapper','Flashlight','ASB','DiabetesM','HealthyRecipes','ChinaDaily','GooglePay','MapMyFitness','Phone','Mobills','Running','SalatTime','Fit','FitBreathe','DailyTracking','Battery']

order = appsByMedianTxRx

[y_test, y_pred] = deserialize("app-id-huaweiwatch-def-delay_group-cm.json")

cm, fig, ax = plot_confusion_matrix(y_test, y_pred, nolabel=False,labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", noTextBox=True, figsize=(9, 8), colorbar=True, classes=order)

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

write_latex_table_precision_recall_f1("app-id-huaweiwatch-def-delay_group-cm.tex", y_test, y_pred)
plt.tight_layout()
plt.savefig('app-id-huaweiwatch-def-delay_group-cm.png', format='png')
plt.savefig('app-id-huaweiwatch-def-delay_group-cm.png'.replace('.png', '.eps'), format='eps')
print("Written app-id-huaweiwatch-def-delay_group-cm.png/eps")
