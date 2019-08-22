# Modifications and Notes for STC

Christian Ritter

HGru4Rec by Quadrana17

### Comments

* summary stats is not written out. needs to be done
* no checkpoints written out...
* Review Top1 loss implementation
* Make variable .is_training in model optional?
* Make variable self.decay_steps = 1e4 optional?
* It is differentiatied between input and output embeddings
* Parameter final_act in model module is not used as input parameter in train_*
* Uses exponential decayed lr with Adam instead of AdaGrad as in paper
* modifications for predictions
    * HGRU4Rec initialization for self.is_training==False needs to be modified

#### Data

Input data is in hdf format, contains `train` and `valid_train` data field.
Default: data/retail_rocket.hdf

