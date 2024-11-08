# Useful commands

```bash
uv sync --extra dev --prerelease allow
```

```bash
cd doc
make.bat html
build\html\index.html
xcopy /c /e /s /h build\html ..\docs
cd ..
```

* Download and extract wheel artifacts from GitHub.

```bash
cd /d "C:\Users\carlk\Downloads\wheels (46)"
twine upload fastlmm*
```

Create a local distribution"

```bash
uv build
twine upload dist/fastlmm-0.????.tar.gz

```

## Old

```bash
python setup.py sdist
twine upload dist/*.tar.gz

conda activate ./.conda
pip install -r requirements.txt

set pythonpath=O:\programs\bed-reader;O:\programs\fastlmm;
set pythonpath=O:\programs\fastlmm


# docs
pip install sphinx
cd doc
make html
build\html\index.html
xcopy /c /e /s /h build\html ..\docs

cd tests
python test.py

conda create --name py11 python=3.11
conda activate py11
conda create --name py8 python=3.8
conda activate py8
conda install numpy scipy
# OK

cd ~/programs/fastlmm
conda create --name py9 python=3.9
conda activate py9
conda install numpy scipy
pip install -r requirements.txt
cd ~/programs/fastlmm/fastlmm/inference/tests
python test.py

cd ~/programs/fastlmm
conda create --name py10 python=3.10
conda activate py10
conda install numpy scipy
pip install -r requirements.txt
cd ~/programs/fastlmm/fastlmm/inference/tests
python test.py

cd ~/programs/fastlmm
conda create --name py11b python=3.11
conda activate py11b
conda install numpy scipy
pip install -r requirements.txt
cd ~/programs/fastlmm/fastlmm/inference/tests
python test.py

cd ~/programs/fastlmm
conda create --name py11c python=3.11
conda activate py11c
pip install scipy==1.10.1
pip install -r requirements.txt
cd ~/programs/fastlmm/fastlmm/inference/tests
python test.py

cd ~/programs/fastlmm
conda create --name py11d python=3.11
conda activate py11d
pip install scipy==1.11.0
pip install -r requirements.txt
cd ~/programs/fastlmm/fastlmm/inference/tests
python test.py

---------
code-insiders .

export PATH=$PATH:/home/carlk/miniconda3/bin
conda activate ./.conda
conda activate py11
pip install -r requirements.txt
export PYTHONPATH=~/programs/fastlmm
cd tests
cd ~/programs/fastlmm/fastlmm/inference/tests
python test.py


while python test.py; do
    :
done

twine upload --repository-url https://test.pypi.org/legacy/ fastlmm-0.6.7*
```
