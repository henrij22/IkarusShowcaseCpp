# SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
# SPDX-License-Identifier: MIT

add_executable("iga_cpp" iga_cpp.cpp)
target_link_dune_default_libraries("iga_cpp")

# Copy the example grid used for testing into the build dir
file(COPY input/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/input)