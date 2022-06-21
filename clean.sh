#!/bin/bash
rm -rf _skbuild
rm -rf _cmake_test_compile
rm -rf kwimage_ext.egg-info
rm -rf pip-wheel-metadata

rm -rf kwimage_ext/structs/_mask_backend/cython_mask*.*so
rm -rf kwimage_ext/structs/_boxes_backend/cython_boxes*.*so
rm -rf kwimage_ext/algo/_nms_backend/*_nms.*so
rm -rf kwimage_ext/structs/_mask_backend/cmake_install.cmake
rm -rf kwimage_ext/structs/_mask_backend/CMakeFiles
rm -rf kwimage_ext/structs/_boxes_backend/cython_boxes.html
rm -rf kwimage_ext/structs/_boxes_backend/cython_boxes.c
rm -rf kwimage_ext/structs/_boxes_backend/cmake_install.cmake
rm -rf kwimage_ext/structs/_boxes_backend/CMakeFiles

rm -rf htmlcov
rm -rf build.ninja
rm -rf cmake_install.cmake

rm -rf kwimage_ext/algo/_nms_backend/cpu_nms.c
rm -rf kwimage_ext/algo/_nms_backend/cpu_nms.cpp
rm -rf kwimage_ext/algo/_nms_backend/gpu_nms.cpp
rm -rf kwimage_ext/algo/_nms_backend/gpu_nms.cxx
rm -rf kwimage_ext/algo/_nms_backend/*.so
rm -rf kwimage_ext/algo/_nms_backend/CMakeFiles
rm -rf kwimage_ext/algo/_nms_backend/cmake_install.cmake
