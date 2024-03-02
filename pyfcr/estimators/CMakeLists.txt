cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(_estimators)

include_directories("src/")

file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "src/*.hpp")
file (GLOB PYTHON_FILES "*.cpp")

find_package(pybind11 REQUIRED)
find_package(Python 3.10 EXACT COMPONENTS Development)
pybind11_add_module(_estimators
	${SOURCE_FILES}
	${HEADER_FILES}
	${PYTHON_FILES}
)

target_compile_definitions(_estimators PRIVATE VERSION_INFO=${VERSION_INFO})
target_include_directories(_estimators PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(_estimators PRIVATE ${Python_LIBRARIES})

target_compile_options(_estimators PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-static-libgcc -static-libstdc++>
    $<$<CXX_COMPILER_ID:GNU>:$<$<BOOL:${MINGW}>:-static>>
)

target_link_options(_estimators PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-static-libgcc -static-libstdc++>
    $<$<CXX_COMPILER_ID:GNU>:$<$<BOOL:${MINGW}>:-static>>
)


message(STATUS "Python Include Directories: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python Libraries: ${PYTHON_LIBRARIES}")
message(STATUS "Python Executable: ${PYTHON_EXECUTABLE}")

