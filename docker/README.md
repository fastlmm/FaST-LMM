## Docker Instructions

To build the docker container, run:

```bash
cd FaST-LMM/docker
docker build -t fastlmm .
```

The container has no entry point but jupyter will be installed so
a common use case for execution is:

```bash
# Mount local repo copy for access to datasets
REPO_DIR=$HOME/repos/FaST-LMM
docker run --rm -p 8888:8888 -it \
-v $REPO_DIR:/work/FaST-LMM \
fastlmm /bin/bash

# In container:
(fastlmm)> jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''

# Use notebook at localhost:8888
```
