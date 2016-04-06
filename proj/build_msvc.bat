@echo off
setlocal
pushd ..
mkdir build >NUL 2>NUL
cd build
cmake -G "Visual Studio 11 2012 Win64" .. && cmake --build . --config Release
