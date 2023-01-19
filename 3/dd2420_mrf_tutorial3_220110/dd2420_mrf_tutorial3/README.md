# KTH PGM Course (DD2420) - Tutorial 3 on MRFs

## Installation

Install conda environment with the following command:
```
conda env create -f environment.yml
```
Environment was created using the command ```conda create -n dd2420_lbp_tutorial3 numpy matplotlib pillow```

Install required packages using pip with the following command:
```
pip install -r requirements.txt
```

## Run code
Run loopy belief propagation on the image denoising task by executing
```
python main.py
```
The original, noisy, and revoered images will be saved in ```./results/```.
