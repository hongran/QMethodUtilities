# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fanghan/GM2/gpuSim/qsim

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fanghan/GM2/gpuSim/qsim/build

# Include any dependencies generated for this target.
include CMakeFiles/TruthAnalysisModule.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TruthAnalysisModule.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TruthAnalysisModule.dir/flags.make

CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o: CMakeFiles/TruthAnalysisModule.dir/flags.make
CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o: ../src/TruthAnalysisModule.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fanghan/GM2/gpuSim/qsim/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/fanghan/GM2/gpuSim/qsim/src/TruthAnalysisModule.cu -o CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o

CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.requires:

.PHONY : CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.requires

CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.provides: CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.requires
	$(MAKE) -f CMakeFiles/TruthAnalysisModule.dir/build.make CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.provides.build
.PHONY : CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.provides

CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.provides.build: CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o


# Object files for target TruthAnalysisModule
TruthAnalysisModule_OBJECTS = \
"CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o"

# External object files for target TruthAnalysisModule
TruthAnalysisModule_EXTERNAL_OBJECTS =

libTruthAnalysisModule.a: CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o
libTruthAnalysisModule.a: CMakeFiles/TruthAnalysisModule.dir/build.make
libTruthAnalysisModule.a: CMakeFiles/TruthAnalysisModule.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fanghan/GM2/gpuSim/qsim/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libTruthAnalysisModule.a"
	$(CMAKE_COMMAND) -P CMakeFiles/TruthAnalysisModule.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TruthAnalysisModule.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TruthAnalysisModule.dir/build: libTruthAnalysisModule.a

.PHONY : CMakeFiles/TruthAnalysisModule.dir/build

CMakeFiles/TruthAnalysisModule.dir/requires: CMakeFiles/TruthAnalysisModule.dir/src/TruthAnalysisModule.cu.o.requires

.PHONY : CMakeFiles/TruthAnalysisModule.dir/requires

CMakeFiles/TruthAnalysisModule.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TruthAnalysisModule.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TruthAnalysisModule.dir/clean

CMakeFiles/TruthAnalysisModule.dir/depend:
	cd /home/fanghan/GM2/gpuSim/qsim/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fanghan/GM2/gpuSim/qsim /home/fanghan/GM2/gpuSim/qsim /home/fanghan/GM2/gpuSim/qsim/build /home/fanghan/GM2/gpuSim/qsim/build /home/fanghan/GM2/gpuSim/qsim/build/CMakeFiles/TruthAnalysisModule.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TruthAnalysisModule.dir/depend

