cmake_minimum_required(VERSION 3.4.3)
project(vulkii)
find_package(Vulkan REQUIRED)
IF (NOT Vulkan_FOUND)
    find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
    IF (Vulkan_LIBRARY)
        set(Vulkan_FOUND ON)
        MESSAGE("Using bundled Vulkan library version")
    ENDIF()
ENDIF()
IF (NOT Vulkan_FOUND)
  message(FATAL_ERROR "Could not find Vulkan library!")
ELSE()
  message(STATUS ${Vulkan_LIBRARY})
  message(STATUS "$ENV{VULKAN_SDK}/include")
ENDIF()
add_executable(vulkii
src/main.cpp
)
target_include_directories(vulkii
  PRIVATE
  3rdparty
  include
  3rdparty/imgui
  "$ENV{VULKAN_SDK}/include"
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
-march=skylake \
-pedantic-errors -Wall -Wextra -Werror -ferror-limit=1 \
-Wno-c99-extensions -Wno-comment \
-fno-exceptions -fno-rtti -fvisibility=hidden")
target_link_libraries(vulkii
"$ENV{VULKAN_SDK}/lib/libSPIRV-Tools-shared.so"
${Vulkan_LIBRARY}
SDL2
pthread
ncurses z dl)
