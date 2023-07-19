# SAJA-Dilepton

## environment setup
```bash
git clone git@github.com:JeewonHeo/SAJA-Dilepton.git
cd SAJA-Dilepton

conda env create -f environment.yaml
conda activate saja-dilep-py311
```

## run tutorial script
```bash
conda activate saja-dilep-py311
cd tutorial
python3 tutorial_train.py
```

## Jupyter setup
```bash
python -m ipykernel install --user --name saja-dilep-py311 --display-name "saja-dilep-py311"
```
after this command you can find the kernel in your jupyter notebook




