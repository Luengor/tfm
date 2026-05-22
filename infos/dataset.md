# Salamanca (for the TFM)
Given in 2 sets, one after the other. Because of this, they serve diffrent purposes:
 - 265 images. Used for training the heads and fine tuning yolo.
    - 110 crops are hand labeled from this dataset for training the heads on the
      style classification task.
 - 5880 + 536 = 6416 images. Full dataset used for evaluation.
    - From this dataset, ~300 crops (294) are used for evaluation of the heads on
      the style classification task.

# Cuenca (stopgrafiti)
A set of 1207 images. Used for trainning the heads and fine tuning yolo.
 - 275 crops are hand labeled from this dataset for training the heads on the
   style classification task. Added up to the 110 from Salamanca, we have 385
   crops for training the style heads.


