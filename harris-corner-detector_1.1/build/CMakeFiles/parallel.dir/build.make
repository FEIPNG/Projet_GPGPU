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
include CMakeFiles/parallel.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/parallel.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/parallel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/parallel.dir/flags.make

CMakeFiles/parallel.dir/src/parallel.cpp.o: CMakeFiles/parallel.dir/flags.make
CMakeFiles/parallel.dir/src/parallel.cpp.o: ../src/parallel.cpp
CMakeFiles/parallel.dir/src/parallel.cpp.o: CMakeFiles/parallel.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/user/7/hajbie/harris-corner-detector_1.1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/parallel.dir/src/parallel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/parallel.dir/src/parallel.cpp.o -MF CMakeFiles/parallel.dir/src/parallel.cpp.o.d -o CMakeFiles/parallel.dir/src/parallel.cpp.o -c /user/7/hajbie/harris-corner-detector_1.1/src/parallel.cpp

CMakeFiles/parallel.dir/src/parallel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/parallel.dir/src/parallel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /user/7/hajbie/harris-corner-detector_1.1/src/parallel.cpp > CMakeFiles/parallel.dir/src/parallel.cpp.i

CMakeFiles/parallel.dir/src/parallel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/parallel.dir/src/parallel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /user/7/hajbie/harris-corner-detector_1.1/src/parallel.cpp -o CMakeFiles/parallel.dir/src/parallel.cpp.s

# Object files for target parallel
parallel_OBJECTS = \
"CMakeFiles/parallel.dir/src/parallel.cpp.o"

# External object files for target parallel
parallel_EXTERNAL_OBJECTS =

parallel: CMakeFiles/parallel.dir/src/parallel.cpp.o
parallel: CMakeFiles/parallel.dir/build.make
parallel: src/libhcd.a
parallel: src/libhcd_cuda.a
parallel: CMakeFiles/parallel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/user/7/hajbie/harris-corner-detector_1.1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable parallel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parallel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/parallel.dir/build: parallel
.PHONY : CMakeFiles/parallel.dir/build

CMakeFiles/parallel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/parallel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/parallel.dir/clean

CMakeFiles/parallel.dir/depend:
	cd /user/7/hajbie/harris-corner-detector_1.1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /user/7/hajbie/harris-corner-detector_1.1 /user/7/hajbie/harris-corner-detector_1.1 /user/7/hajbie/harris-corner-detector_1.1/build /user/7/hajbie/harris-corner-detector_1.1/build /user/7/hajbie/harris-corner-detector_1.1/build/CMakeFiles/parallel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/parallel.dir/depend

