# PoisonPy Dataset

This directory contains the **PoisonPy** dataset organized as follows:

The ``Baseline Training Set`` folder contains a .json file with the entire clean training set (i.e., without any data poisoning). The .json file contains the following fields:
1. *text*: the NL code description;
2. *code*: the Python code snippet implementing the intended description;
3. *vulnerable*: indicating whether the code snippet is safe (0) or unsafe (1);
4. *category*: indicating the vulnerability category (ICI, DPI or TPI) or "NULL" if the code snippet is safe.

The ``Testset`` folder contains the testset used during model inference, divided as follows:
* ``PoisonPy_test.in``, containing the intents of the test set; 
* ``PoisonPy_test.out``, containing the code snippets of the test set.

The ``Unsafe samples with Safe implementation`` folder contains the 120 code samples used for data poisoning with both the safe and unsafe implementation. There are 40 samples belonging to each category, i.e., ICI, DPI and TPI.
* The ``120_clean.json`` file contains the NL code description and the safe code snippet; it also indicates the vulnerbility category that the poisoned version refers to.
* The ``120_poisoned.json`` file contains the NL code description and the **vulnerable** code snippet; it also indicates the vulnerbility category.


# Code for the Targeted Data Poisoning Attack

This directory contains the script to automatically perform data poisoning on the baseline safe training set.

The README file is written based on our setup experience on *Ubuntu 18.04.3 LTS*. 

To run the code, use ``python data_poisoning_attack.py [VULN_CATEG] [N]``.

The script takes 2 arguments: 
1. [VULN_CATEG] Vulnerability category ("ICI", "DPI", or "TPI")
2. [N], Number of samples to poison (5, 10, 15, ..., 40)

Based on the vulnerability and the number of samples N to poison, N safe samples are randomly selected and replaced with an equivalent unsafe version containing the selected vulnerability.

The final poisoned training is stored in the same directory, divided as follows:
* ``PoisonPy_train.in``, containing the intents of the training set; 
* ``PoisonPy_train.out``, containing the code snippets of the training set; 
* ``PoisonPy_dev.in``, containing the intents of the validation set;  
* ``PoisonPy_dev.out``, containing the code snippets of the validation set.

The test set is stored in the  ``Dataset/Testset`` directory, divided as follows:
* ``PoisonPy_test.in``, containing the intents of the test set; 
* ``PoisonPy_test.out``, containing the code snippets of the test set.