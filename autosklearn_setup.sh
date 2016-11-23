sudo apt-get --yes update
sudo apt-get --yes upgrade
sudo apt-get --yes install make
sudo apt-get --yes install wget
sudo apt-get --yes install git
sudo apt-get --yes install screen
sudo apt-get --yes install python-setuptools
sudo apt-get --yes install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config
sudo apt-get --yes install default-jre
sudo apt-get --yes install libblas-dev liblapack-dev libatlas-base-dev gfortran
#Install necessary python packages
sudo apt-get --yes install python-pip
sudo pip install virtualenv
virtualenv venv_autosklearn
source venv_autosklearn/bin/activate
pip install numpy
pip install scipy
pip install xmltodict
sudo apt-get --yes install build-essential
sudo apt-get --yes install python-dev
pip install scikit-learn==0.16.1

git clone https://github.com/jaberg/skdata.git
cd skdata && python setup.py install
cd $HOME

git clone https://github.com/automl/auto-sklearn.git
cd auto-sklearn && git reset --hard 39974ba42c18506b1abd7b7efc51d58ba6258959
pip install -r requ.txt
python setup.py install
cd $HOME

git clone https://github.com/mula0513/pyMetaLearn.git
cd pyMetaLearn
python setup.py install 
cd $HOME

git clone -b autosklearn https://github.com/mula0513/hyperband2.git
cd hyperband2 && python setup.py install
cd $HOME
mv hyperband2 HPOlib

