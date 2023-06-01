# DRGDL
This is a public code for predicting drug-disease associations

### 'data' directory
Contain C-dataset and F-dataset.

### negativesamples
To produce negativesamples for DDA prediction by DRGDL
  - python NegativeSamples.py

### main.py
To predict drug-disease associations by DRGDL, run
  - python main.py 
  - -d is dataset selection, including C-dataset and F-dataset.

### Options
See help for the other available options to use with *DRGDL*
  - python main.py --help

### Requirements
DRGDL is tested to work under Python 3.6.0+  
The required dependencies for DRGDL are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.

### Contacts
If you have any questions or comments, please feel free to email BoWei Zhao (stevejobwes@gmail.com).
