# AI Engineering Project - Usefu Commands

##### 1. Virtual Environment

Activate uv environment:
```bash
uv run bash  # 
source .venv/bin/activate  # classic activation
deactivate # deactivate

```
##### 2. Packages
```bash
uv pip list # check which packages
uv pip list | grep package_name
# install stuffs
uv add package_name==version
uv sync        # installs all dependencies from pyproject.toml
```

##### 3. Jupyter
```bash
# start jupyter notebook
uv run jupyter notebook
uv run jupyter lab
# register uv environment as a kernel:
uv add ipykernel # dependenciy if no present
uv run python -m ipykernel install --user --name=ai_engineering --display-name "Python (ai_engineering)"
# list available kernels
jupyter kernelspec list
# remove an old kernel
jupyter kernelspec uninstall old_kernel_name
```