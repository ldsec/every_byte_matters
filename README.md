# Every Byte Matters: Traffic Analysis of Wearable Devices

This repository contains the code and the plots for the inference attack described in the paper `Every Byte Matters: Traffic Analysis of Wearable Devices`, L. Barman, A. Dumur, A. Pyrgelis, and J.-P. Hubaux, published at Ubicomp 2021. [Link to the paper](every_byte_matters_traffic_analysis_wearable_devices.pdf)

Maintainer: [Ludovic Barman](https://people.epfl.ch/ludovic.barman)

## How to reproduce

Prerequisites: `python3` and `pip`.

Step 1: We make the dataset available for research purposes; please contact us to obtain a copy.

Step 2: Install the requirements via `pip install -r requirements.txt`

Step 3: Edit `constants.py` and point to the dataset directory (which must be downloaded separately)

Step 4: Run one of the following files. The section numbering correspond to the sections in the paper.

- `device_id.py`: §5 device identification
- `chipset_id.py`: §5 chipset identification
- `action_id_wearables.py`: §6.1 "wide" experiment on action recognition, all wearable devices
- `app_id_huaweiwatch.py`: $6.2 "deep" experiment on WearOS, application-opening recognition
- `app_id_transfer.py`: §6.2.2 transfer experiment between different pairs of wearable devices
- `action_id_diabetesm.py`: §6.2.3 recognizing fine-grained actions within DiabetesM
- `longrun.py`: §6.2.4 long-term adversary
- `aging_training_{1,3}day.py`: §6.2.5 aging of the dataset
- `packet_loss_app_id_huaweiwatch.py`: §8 impact of packet loss

### Technical details

Under the hood, each one of the above files is organised in the same way:

- Initially, the dataset is parsed and cached. There are two levels of caching to make the thing fast: a per-file caching, and a global cache per attack. Everything is cached into a folder `.cache`. Removing the cache folder triggers a deep-rebuild with the per-file caches (this should never be needed, unless the CSV parsing changes). Otherwise, there is a flag `REBUILD=True` to only rebuild the high-level cache file for the current attack. 
- Then, each trace is mapped to the appropriate label + features via `build_features_labels_dataset()` 
- Finally, Random Forest is run, creating plots. Each plot is saved with its data + the git commit of the dataset, in the `PLOT_NAME.json` file; in addition, there is a latex table generated as `PLOT_NAME.tex` in addition to the graphical output `PLOT_NAME.{eps,png}`

## Figures used in the paper

All figures are in `inference_attack/plots/`

- Fig 4a: `device-id-cla-cm`
- Fig 4b: `device-id-ble-cm`
- Fig 5a: `device-id-cla-fi`
- Fig 5b: `device-id-ble-fi`
- Fig 6a: `action-id-wearables-cm`
- Fig 6b: `action-id-wearables-fi`
- Fig 7a: `app-id-huaweiwatch-cm`
- Fig 7b: `app-id-huaweiwatch-fi`
- Fig 8a: `action-id-diabetesm-cm`
- Fig 8b: `action-id-diabetesm-fi`
- Fig 9: `longrun_p_r_f1_threshold`
- Fig 10a: `aging`
- Fig 10b: `aging_per_class`

## How to cite

To be updated

## Acknowledgements

This work has been made possible by the help of Friederike Groschupp and Stéphanie Lebrun.
We also wish to thank Daniele Antonioli, Sylvain Chatel, Jiska Classen, Ricard Delgado, and David Lazar for the constructive discussions and feedbacks on the drafts.
We are grateful to the ``Centre Suisse d'Electronique et Microtechnique'' (CSEM) for providing us with the Ellisys Bluetooth sniffer.
This work was supported in part by grant 200021\_178978/1 (PrivateLife) of the Swiss National Science Foundation (SNF).
Some illustrations in the paper have been made by the artists freepik, eucalyp and smashicons ([flaticon.com](flaticon.com)).