###############################################################################
#
# Build script for project
#
###############################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)


#set(CCFILES
#  scan_gold.cpp
#  )

set(CUFILES
  stringSortCUDPP.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})

#cuda_add_executable(stringSortCUDPP ${CCFILES} ${CUFILES})

cuda_add_executable(stringSortCUDPP ${CUFILES})

target_link_libraries(stringSortCUDPP cudpp rt)
