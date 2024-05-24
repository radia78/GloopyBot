# setup.sh
#!/bin/bash

echo "Updating the packages"
sudo apt-get update

echo "Installing pip"
sudo apt install -y python3-pip

echo "Running setup script"
sudo pip3 install -r requirements.txt
