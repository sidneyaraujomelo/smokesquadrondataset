# Smoke Squadron Dataset

This repository contains the Smoke Squadron Dataset, a game provenance graphs dataset composed of 37 game sessions recorded using the PinG Framework for Game Analytics and Machine Learning tasks. Currently, we provide both the raw provenance graphs in the "original provenance graphs" folder and a framework agnostic version of the dataset in the "ss_het" folder.

## Original Provenance Graphs

The folder contains 37 xml files recorded directly from Smoke Squadron's game sessions. The XML is defined by the PinG Framework and was used to create the "ready to use" data in "ss_het".

## SS_HET

The folder contains several files, which we explain in detail:
* **prov-G.json**: NX graph data.
* **prov-atbset_list.json**: Lists the feature sets of each node type.
* **provatbset_N-feats.npy** : Feature matrix for node type N, where each line corresponds to a local node index.
* **provatbset_N-map.json** : Json dictionary mapping local node index of node type N to node index in prov-G.json.
