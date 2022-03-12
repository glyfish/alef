# alef

The first version

## pyenv

MAC OS

```
pyenv install 3.9.2
pyenv virtualenv 3.9.2 gly.fish.3.9.2
pyenv activate gly.fish.3.9.2
```

Ubuntu

```
pyenv install 3.6.1
pyenv virtualenv 3.6.1 gly.fish
pyenv activate gly.fish
```

## Install Packages

MAC OS

```
cat packages.txt | xargs pip install
```
or

```
pip install -r requirements-macos.txt
```

Ubuntu

```
cat packages.txt | xargs pip install
```

```
pip install -r requirements-ubuntu.txt
```

