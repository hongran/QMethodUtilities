cmake_minimum_required(VERSION 3.10)

# set the project name
project(qsim CUDA CXX)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ROOT REQUIRED)
find_package(CUDA)

include(${ROOT_USE_FILE})
include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# add the executable
add_executable(QSimulation src/QSimulation.cpp src/QSim.cu src/AnalyzerRegistration.cu src/AnalysisModuleBase.cu)
target_sources(QSimulation PUBLIC 
  src/RHAnalysisModule.cu 
  src/RPAnalysisModule.cu 
  src/TruthAnalysisModule.cu 
  src/EnergyHistogramModule.cu
  src/FFTAnalysisModule.cu
  ) 
target_link_libraries(QSimulation ${ROOT_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES}) 

