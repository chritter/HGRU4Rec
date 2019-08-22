# Modifications and Notes for STC

Christian Ritter

* HGru4Rec by Quadrana17, implementation of https://github.com/jangmino/HGRU4Rec

#### Comments

* summary stats is not written out. needs to be done
* Review Top1 loss implementation
* Make variable self.decay_steps = 1e4 optional?
* It is differentiatied between input and output embeddings
* Parameter final_act in model module is not used as input parameter in train_*
* Uses exponential decayed lr with Adam instead of AdaGrad as in paper
* I uncommented the code line to save checkpoints in model.py

#### Input Data

Input data is in hdf format, contains `train` and `valid_train` data field.
Default: data/retail_rocket.hdf

#### What needs to be done

* Make variable .is_training in model optional?
* Implement metric to evaluate Hit@k or MRR as in paper
* modifications for evaluation with validation data
    * iterate should include 
    * Make sure training data has 'valid_train' data in hdf file, currently not available
* modifications for predictions
    * there needs to be done some modifications to allow predictions (besides the standard evaluation during the traing)
    * HGRU4Rec initialization for self.is_training==False needs to be modified to load checkpoint. See commented code


