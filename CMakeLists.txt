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
IF (WIN32)
  if(EXISTS "${SDL2_PATH}/lib/x64/SDL2.lib")
  else()
	message(FATAL_ERROR "Please provide path to SLD2! e.g. -D SDL2_PATH=C:/SDL2")
  endif()
  add_custom_command(TARGET vulkii POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2_PATH}/lib/x64/SDL2.dll"
        $<TARGET_FILE_DIR:vulkii>)
  set (SPIRV_TOOLS_LIB "$ENV{VULKAN_SDK}/lib/SPIRV-Tools-shared.lib")
  set (LIBS "${SDL2_PATH}/lib/x64/SDL2.lib" "${SDL2_PATH}/lib/x64/SDL2main.lib")
  set (INCLUDES "${SDL2_PATH}/include")
ELSE()
  set (SPIRV_TOOLS_LIB "$ENV{VULKAN_SDK}/lib/libSPIRV-Tools-shared.so")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
	-march=skylake \
	-pedantic-errors -Wall -Wextra -Werror -ferror-limit=1 \
	-Wno-c99-extensions -Wno-comment \
	-fno-exceptions -fno-rtti -fvisibility=hidden")
  set (LIBS pthread ncurses z dl)
  set (INCLUDES "")
ENDIF()
target_include_directories(vulkii
  PRIVATE
  include
  ${INCLUDES}
  "$ENV{VULKAN_SDK}/include"
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
target_link_libraries(vulkii
${SPIRV_TOOLS_LIB}
${Vulkan_LIBRARY}
${LIBS}
)
