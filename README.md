###README File that explains how to load in the training , validation and test sets of the ###runs on Red , Green , Blue and Grayscale color images 
##########################################################
###   INSTRUCTIONS 
##########################################################


1 GRAYSCALE :
  a) TRAINING
   The grayscale image training set is named as "da_hidden_vals_8004_original.h5 ". It contains the hidden layer representations of the training set ( 4000 images ) run on a one layer deep autoencoder . 
Open it using the following : 
>>import h5py 
>>f=h5py.File("da_hidden_vals_8004_original.h5")
>>training_set_hidden=  f["hidden"][:] 

This is a 4000*500 sized numpy array 


  b) VALIDATION
    The grayscale image validation set has its hidden representation named as "valid_set_hidden_rep_gray.h5" . You must read it in the following way : 
>>import h5py 
>>f=h5py.File("valid_set_hidden_rep_gray.h5")
>>valid_set_hidden= f["valid_set_hidden"][:] 
 
This is a 2000*500 sized numpy array 


  c) TEST 
   The grayscale image test set has its hidden representation named as "test_set_hidden_rep.h5" . It contains the hidden representations of the test set images . Load it using the following mechanism 
>>import h5py 
>>f=h5py.File("test_set_hidden_rep.h5")
>>test_set_hidden = f["training_set_hidden"][:]
Please note that this key called training_set_hidden was used by mistake for the test set . Please continue using this 


2. COLOR CHANNELS 
  Colors : Red , Green , Blue 

   a) TRAINING 
    The training_set_hidden reps are named as "training_set_hidden_rep_<COLOR>".h5 where you can replace color with  "red" , "green" , "blue" . So for example the red color channel training set hidden reps should be opened as : 
 >> import h5py 
 >> f=h5py.File("training_set_hidden_rep_red.h5") 
 >>training_set_hidden =  f["training_set_hidden"][:] 

   b) VALIDATION 
 >>import h5py 
 >>f=h5py.File("valid_set_hidden_rep_red.h5")
 >>valid_set_hidden =  f["valid_set_hidden"][:] 
   c) TESTING 
 >>import h5py 
 >>f=h5py.File("test_set_hidden_rep_red.h5") 
 >>test_set_hidden_rep = f["test_set_hidden"][:] 

##TODO : Update more as more code is added
