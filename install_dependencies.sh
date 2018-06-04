#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################


TOPDIR=$PWD

# Prefix:
PREFIX=$PWD/torch
echo "Installing Torch into: $PREFIX"

if [[ `uname` != 'Linux' ]]; then
  echo 'Platform unsupported, only available for Linux'
  exit
fi
if [[ `which apt-get` == '' ]]; then
    echo 'apt-get not found, platform not supported'
    exit
fi

# Install dependencies for Torch:
sudo apt-get update
sudo apt-get install -qqy build-essential
sudo apt-get install -qqy gcc g++
sudo apt-get install -qqy cmake
sudo apt-get install -qqy curl
sudo apt-get install -qqy libreadline-dev
sudo apt-get install -qqy git-core
sudo apt-get install -qqy libjpeg-dev
sudo apt-get install -qqy libpng-dev
sudo apt-get install -qqy ncurses-dev
sudo apt-get install -qqy imagemagick
sudo apt-get install -qqy unzip
sudo apt-get install -qqy libqt4-dev
sudo apt-get install -qqy liblua5.1-0-dev
sudo apt-get install -qqy libgd-dev
sudo apt-get update

#rewrite if neded, this is needed due to broken torch.inverse with cuda tensors
export CUDADIR=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin # for NVCC
#assuming openblas is used, folow magma install guid otherwise 
export OPENBLASDIR=/opt/OpenBLAS
wget 'http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.3.0.tar.gz'
tar -xvsf magma-2.3.0.tar.gz
cd magma-2.3.0
cp make.inc-examples/make.inc.openblas make.inc
make
sudo mkdir -p /opt/magma
sudo mkdir -p /usr/local/magma
USER_=`whoami`
sudo chown $USER_ /usr/local/magma
make install
sudo chown root /usr/local/magma
echo "==> Torch7's dependencies have been installed"





# Build and install Torch7
cd /tmp
rm -rf luajit-rocks
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make install
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi

path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    cutorch=ok
    cunn=ok
fi

# Install base packages:
$PREFIX/bin/luarocks install cwrap
$PREFIX/bin/luarocks install paths
$PREFIX/bin/luarocks install torch
$PREFIX/bin/luarocks install nn

[ -n "$cutorch" ] && \
($PREFIX/bin/luarocks install cutorch)
[ -n "$cunn" ] && \
($PREFIX/bin/luarocks install cunn)

$PREFIX/bin/luarocks install luafilesystem
$PREFIX/bin/luarocks install penlight
$PREFIX/bin/luarocks install sys
$PREFIX/bin/luarocks install xlua
$PREFIX/bin/luarocks install image
$PREFIX/bin/luarocks install env
$PREFIX/bin/luarocks install qtlua
$PREFIX/bin/luarocks install qttorch
$PREFIX/bin/luarocks install optim

echo ""
echo "=> Torch7 has been installed successfully"
echo ""


echo "Installing nngraph ... "
$PREFIX/bin/luarocks install nngraph
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "nngraph installation completed"


echo "Installing Lua-GD ... "
mkdir $PREFIX/src
cd $PREFIX/src
rm -rf lua-gd
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile
$PREFIX/bin/luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Lua-GD installation completed, attach torch/bin directory to env path"

echo "For the first time please download and extract the pretrained w2v from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/ to Action-Eliminating-DQN/dqn/"
echo 
echo "You can run experiments by executing: "
echo
echo "   ./run_gpu zork <scenario> <agent_type> [GPU_ID]"

