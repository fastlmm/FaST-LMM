conda activate ./.conda
pip install -r requirements.txt

set pythonpath=O:\programs\bed-reader;O:\programs\fastlmm;O:\programs\pysnptools
set pythonpath=O:\programs\fastlmm

cd tests
python test.py

conda create --name py11 python=3.11
conda activate py11

---------
code-insiders .

export PATH=$PATH:/home/carlk/miniconda3/bin
conda activate ./.conda
pip install -r requirements.txt
export PYTHONPATH=~/programs/fastlmm
cd tests
python test.py
