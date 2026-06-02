# Salamanca (for the TFM)
Given in 2 sets (6681 images in total), one after the other. Because of this,
they serve different purposes:
 - 265 images. Used for training the heads and fine tuning yolo.
    - 110 crops are hand labeled from this dataset for training the heads on the
      style classification task.
    - 185 crops are hand labeled for evaluation of the author head.
 - 5880 + 536 = 6416 images. Full dataset used for evaluation.
    - From this dataset, ~300 crops (294) are used for evaluation of the heads on
      the style classification task.

# Cuenca (stopgrafiti)
A set of 1106 images. Used for trainning the heads and fine tuning yolo.
 - 275 crops are hand labeled from this dataset for training the heads on the
   style classification task. Added up to the 110 from Salamanca, we have 385
   crops for training the style heads.
 - 321 crops are hand labeled for trainning of the author head. 

- 447 images are used for trainning/validation of the yolo model.

