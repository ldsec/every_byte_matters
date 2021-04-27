DATASET_DIR = 'wearables_dataset/data/' # available upon request
AGING_DATA_DIR = DATASET_DIR.replace('/data/', '/data_aux/') +'capture_over_32_days/'

PLOT_DIR = 'plots/'
TEST_PERCENTAGE = 0.2
DEVICES_CLASSIC = ['SamsungGalaxyWatch', 'FossilExploristHR', 'AppleWatch', 'Airpods', 'MDR', 'HuaweiWatch2', 'FitbitVersa2'] # other devices are BLE
N_FOLDS = 10

FILTER_APP_MINIMUM_PAYLOAD = 500
FILTER_APP_RATIO_OF_EVENTS_NEEDED = 0.25 # need >25% of samples >= 500B to consider this app+action