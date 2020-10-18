cmake_minimum_required(VERSION 3.10)

# set the project name
project(qsim CUDA CXX)
find_package(ROOT REQUIRED)
find_package(CUDA)
include(${ROOT_USE_FILE})
include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# add the executable
add_executable(QSimulation src/QSimulation.cpp src/AnalyzerRegistration.cu src/AnalysisModuleBase.cu src/RHAnalysisModule.cu src/RPAnalysisModule.cu src/TruthAnalysisModule.cu src/QSim.cu) 
# add_library(AnalyzerRegistration  src/AnalyzerRegistration.cu)
# add_library(AnalysisModuleBase  src/AnalysisModuleBase.cu)
# add_library(RHAnalysisModule OBJECT src/RHAnalysisModule.cu)
# add_library(RPAnalysisModule OBJECT src/RPAnalysisModule.cu)
# add_library(TruthAnalysisModule OBJECT src/TruthAnalysisModule.cu)
# add_library(QSim OBJECT src/QSim.cu)
# set_target_properties(AnalyzerRegistration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(AnalysisModuleBase PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(QSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(QSimulation PROPERTIES ENABLE_EXPORTS ON)

# target_link_libraries(QSimulation ${ROOT_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES} AnalyzerRegistration AnalysisModuleBase QSim RHAnalysisModule RPAnalysisModule TruthAnalysisModule) 
# target_link_libraries(QSimulation ${ROOT_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES} AnalyzerRegistration AnalysisModuleBase QSim RHAnalysisModule RPAnalysisModule TruthAnalysisModule) 
target_link_libraries(QSimulation ${ROOT_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES}) 