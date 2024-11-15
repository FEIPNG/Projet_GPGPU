# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /user/7/hajbie/harris-corner-detector_1.1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /user/7/hajbie/harris-corner-detector_1.1/build

# Include any dependencies generated for this target.
include src/CMakeFiles/hcd_cuda.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/hcd_cuda.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/hcd_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/hcd_cuda.dir/flags.make

src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o: src/CMakeFiles/hcd_cuda.dir/flags.make
src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o: ../src/gradient_gpu.cu
src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o: src/CMakeFiles/hcd_cuda.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/user/7/hajbie/harris-corner-detector_1.1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o"
	cd /user/7/hajbie/harris-corner-detector_1.1/build/src && /usr/local/cuda-11/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o -MF CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o.d -x cu -c /user/7/hajbie/harris-corner-detector_1.1/src/gradient_gpu.cu -o CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o

src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o: src/CMakeFiles/hcd_cuda.dir/flags.make
src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o: ../src/autocorrelation_gpu.cu
src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o: src/CMakeFiles/hcd_cuda.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/user/7/hajbie/harris-corner-detector_1.1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o"
	cd /user/7/hajbie/harris-corner-detector_1.1/build/src && /usr/local/cuda-11/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o -MF CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o.d -x cu -c /user/7/hajbie/harris-corner-detector_1.1/src/autocorrelation_gpu.cu -o CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o

src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target hcd_cuda
hcd_cuda_OBJECTS = \
"CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o" \
"CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o"

# External object files for target hcd_cuda
hcd_cuda_EXTERNAL_OBJECTS =

src/libhcd_cuda.a: src/CMakeFiles/hcd_cuda.dir/gradient_gpu.cu.o
src/libhcd_cuda.a: src/CMakeFiles/hcd_cuda.dir/autocorrelation_gpu.cu.o
src/libhcd_cuda.a: src/CMakeFiles/hcd_cuda.dir/build.make
src/libhcd_cuda.a: src/CMakeFiles/hcd_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/user/7/hajbie/harris-corner-detector_1.1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA static library libhcd_cuda.a"
	cd /user/7/hajbie/harris-corner-detector_1.1/build/src && $(CMAKE_COMMAND) -P CMakeFiles/hcd_cuda.dir/cmake_clean_target.cmake
	cd /user/7/hajbie/harris-corner-detector_1.1/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hcd_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/hcd_cuda.dir/build: src/libhcd_cuda.a
.PHONY : src/CMakeFiles/hcd_cuda.dir/build

src/CMakeFiles/hcd_cuda.dir/clean:
	cd /user/7/hajbie/harris-corner-detector_1.1/build/src && $(CMAKE_COMMAND) -P CMakeFiles/hcd_cuda.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/hcd_cuda.dir/clean

src/CMakeFiles/hcd_cuda.dir/depend:
	cd /user/7/hajbie/harris-corner-detector_1.1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /user/7/hajbie/harris-corner-detector_1.1 /user/7/hajbie/harris-corner-detector_1.1/src /user/7/hajbie/harris-corner-detector_1.1/build /user/7/hajbie/harris-corner-detector_1.1/build/src /user/7/hajbie/harris-corner-detector_1.1/build/src/CMakeFiles/hcd_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/hcd_cuda.dir/depend

