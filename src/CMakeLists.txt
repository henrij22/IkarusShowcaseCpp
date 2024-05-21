# SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
# SPDX-License-Identifier: MIT

add_executable("iga_cpp" iga_cpp.cpp)
target_link_dune_default_libraries("iga_cpp")

add_dune_pythonlibs_flags(iga_cpp)
find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
target_link_libraries(iga_cpp PUBLIC ${PYTHON_LIBRARIES})


# Copy the example grid used for testing into the build dir
file(COPY input/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/input)