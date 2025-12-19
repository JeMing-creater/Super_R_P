## requirements
WSI requires the CLAM library for slicing. The following describes how to install it.
```
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM

conda env create -f env.yml
```
After installing the CLAM environment, install other packages in this environment.
```
conda activate clam_latest
pip install -r requirements.txt
```