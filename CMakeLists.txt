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
add_subdirectory(3rdparty/meshoptimizer)
add_executable(vulkii
src/main.cpp
src/vk_backend.cpp
src/files.cpp
3rdparty/imgui/imgui.cpp
3rdparty/imgui/imgui_widgets.cpp
3rdparty/imgui/imgui_draw.cpp
3rdparty/imgui/imgui_demo.cpp
3rdparty/imgui/examples/imgui_impl_sdl.cpp
)
IF (WIN32)
  set(SDL2_PATH ${CMAKE_SOURCE_DIR}/bin/SDL2)
  if(EXISTS "${SDL2_PATH}/lib/x64/SDL2.lib")
  else()
	message(FATAL_ERROR "Please provide path to SLD2! e.g. -D SDL2_PATH=C:/SDL2")
  endif()
  add_custom_command(TARGET vulkii POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2_PATH}/lib/x64/SDL2.dll"
        $<TARGET_FILE_DIR:vulkii>)
  set (LIBS "${SDL2_PATH}/lib/x64/SDL2.lib"
    "${SDL2_PATH}/lib/x64/SDL2main.lib"
    "$ENV{VULKAN_SDK}/lib/SPIRV-Tools-shared.lib"
    "$ENV{VULKAN_SDK}/lib/shaderc_shared.lib"
    "$ENV{VULKAN_SDK}/lib/spirv-cross-core.lib"
    )
  set (INCLUDES "${SDL2_PATH}/include")
  set_property(TARGET vulkii PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  set(CompilerFlags
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      )
  foreach(CompilerFlag ${CompilerFlags})
    string(REPLACE "/MDd" "/MD" ${CompilerFlag} "${${CompilerFlag}}")
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /GR- /EHs- /EHc- /D _HAS_ITERATOR_DEBUGGING=0 /D _ITERATOR_DEBUG_LEVEL=0")
ELSE()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
	-march=skylake \
	-pedantic-errors -Wall -Wextra -Werror -ferror-limit=1 \
	-Wno-c99-extensions -Wno-comment \
	-fno-exceptions -fno-rtti -fvisibility=hidden")
  set (LIBS SDL2 pthread ncurses z dl
    "$ENV{VULKAN_SDK}/lib/libSPIRV-Tools-shared.so"
    "$ENV{VULKAN_SDK}/lib/libshaderc_shared.so"
    "$ENV{VULKAN_SDK}/lib/libspirv-cross-core.so"
    )
  set (INCLUDES "")
ENDIF()
target_include_directories(vulkii
  PRIVATE
  3rdparty
  3rdparty/imgui
  include
  ${INCLUDES}
  "$ENV{VULKAN_SDK}/include"
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
target_link_libraries(vulkii
${Vulkan_LIBRARY}
meshoptimizer
${LIBS}
)
