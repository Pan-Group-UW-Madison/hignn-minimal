cmake_minimum_required(VERSION 3.16)
project(hignn-minimal LANGUAGES C CXX)

# find dependencies
set (PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Torch REQUIRED)
find_package(Kokkos REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS off)

add_subdirectory(./extern/pybind11)

include_directories(
    include
    ${MPI_INCLUDE_PATH}
)

if (USE_GPU)
    add_compile_definitions(USE_GPU)
    add_compile_options(-O3 -Werror -Wall -Wextra -DNDEBUG --expt-relaxed-constexpr)
else()
    add_compile_options(-O3 -Werror -Wall -Wextra -DNDEBUG)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(hignn SHARED ${sources} interface.cpp)

set_target_properties(hignn PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(hignn SYSTEM PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hignn Kokkos::kokkos)
target_link_libraries(hignn "${TORCH_LIBRARIES}")
target_link_libraries(hignn "${MPI_CXX_LIBRARIES}")
target_link_libraries(hignn pybind11::pybind11)