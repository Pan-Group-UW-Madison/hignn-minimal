import torch
import os
import argparse
import subprocess

if __name__ == '__main__':
    source_path = os.getcwd()
    
    os.system('clear')
    
    os.system('export CXX=$Kokkos_PATH/bin/nvcc_wrapper && cmake -D CMAKE_PREFIX_PATH="$LibTorch_PATH/share/cmake/;$Kokkos_PATH" -D CMAKE_CXX_EXTENSIONS=Off -D USE_GPU:BOOL=On .')
    
    os.system('make -j 4')
    
    os.system('mv libhignn.so hignn.so')