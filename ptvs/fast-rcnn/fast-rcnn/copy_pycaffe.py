"""
This module copies a precompiled caffe python library to 
the caffe-fast-rcnn folder
"""
import shutil
import os
from glob import glob
from os import path as osp

INSTALL_ROOT = osp.join(os.environ['CMATE_INSTALL_ROOT'], 'Default',
                        'Visual Studio 12 2013 Win64', 'Install')

def main():
    """
    This is the main method to copy the files
    """
    # Set the path to your pycaffe here
    source = osp.join(INSTALL_ROOT, 'Caffe', 'python')
    destination = '../../../caffe-fast-rcnn/python'

    shutil.rmtree(destination)
    shutil.copytree(source, destination)

    # also copy the openblas dlls
    source = osp.join(INSTALL_ROOT, 'OpenBLAS', 'bin')
    openblas_dlls = glob(osp.join(source, '*.dll'))
    for _dll in openblas_dlls:
        shutil.copy2(_dll, osp.join(destination, 'caffe'))

if __name__ == "__main__":
    main()
