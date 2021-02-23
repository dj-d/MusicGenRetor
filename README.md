[![SonarCloud](https://github.com/dj-d/MusicGenRetor/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/dj-d/MusicGenRetor/actions/workflows/sonarcloud.yml)

# MusicGenRetor
Music genre classification with Machine Learning

### Download testing music
Run __musicWgetter.sh__ in scripts folder (*this script download testing music sorted my genres*).

Supported genres:
 - Blues
 - Classical
 - Electronic
 - Jazz
 - Pop
 - Rock

Install musicWgetter.sh dependencies:

```
# Debian based
sudo apt-get install parallel

# Arch Linux based
sudo pacman -S parallel
```

Start musicWgetter.sh:

```
cd scripts/
sudo chmod +x musicWgetter.sh
./musicWgetter.sh
```
