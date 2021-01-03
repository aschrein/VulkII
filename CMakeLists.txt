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

IF (WIN32)
  set(SDL2_PATH ${CMAKE_SOURCE_DIR}/bin/SDL2)
  set(IMPLS
    src/vk_backend.cpp
    src/dx12_backend.cpp)
  if(EXISTS "${SDL2_PATH}/lib/x64/SDL2.lib")
  else()
  message(FATAL_ERROR "Please provide path to SLD2! e.g. -D SDL2_PATH=C:/SDL2")
  endif()
  set (LIBS "${SDL2_PATH}/lib/x64/SDL2.lib"
    "${SDL2_PATH}/lib/x64/SDL2main.lib"
    "$ENV{VULKAN_SDK}/lib/SPIRV-Tools-shared.lib"
    "$ENV{VULKAN_SDK}/lib/shaderc_shared.lib"
    "$ENV{VULKAN_SDK}/lib/spirv-cross-core.lib"
    "$ENV{VULKAN_SDK}/lib/HLSL.lib"
    "d3d12.lib"
    "d3dcompiler.lib"
    "dxcompiler.lib"
    "dxgi.lib"
    )
  set (INCLUDES "${SDL2_PATH}/include")
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
  set(IMPLS
    src/vk_backend.cpp)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -march=skylake \
  -ferror-limit=1 -fsanitize=thread \
  -Wno-c99-extensions -Wno-comment \
  -fno-exceptions -fno-rtti -fvisibility=hidden -fms-extensions")
  set (LIBS SDL2 pthread ncurses z dl
    "$ENV{VULKAN_SDK}/lib/libSPIRV-Tools.a"
    "$ENV{VULKAN_SDK}/lib/libspirv-cross-core.a"
    "$ENV{VULKAN_SDK}/lib/libspirv-cross-c-shared.so"
    "$ENV{VULKAN_SDK}/lib/libshaderc_shared.so"
    "libX11-xcb.so"
    )
  set (INCLUDES "")
ENDIF()

add_compile_definitions(GLM_FORCE_SWIZZLE)

function(ADD_EXAMPLE example_name example_file)

    file(GLOB INCLUDES_SRC
    include/*.h
    )
    add_executable(${example_name}
    ${example_file}
    ${IMPLS}
    src/files.cpp
    3rdparty/imgui/imgui.cpp
    3rdparty/imgui/imgui_widgets.cpp
    3rdparty/imgui/imgui_draw.cpp
    3rdparty/imgui/imgui_demo.cpp
    3rdparty/marching_cubes/marching_cubes.cpp
    ${INCLUDES_SRC}
    3rdparty/imgui/examples/imgui_impl_sdl.cpp
    )
    IF (WIN32)
      add_custom_command(TARGET ${example_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDL2_PATH}/lib/x64/SDL2.dll"
            $<TARGET_FILE_DIR:${example_name}>)
     add_custom_command(TARGET ${example_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/bin/dxil.dll"
            $<TARGET_FILE_DIR:${example_name}>)
      set_property(TARGET ${example_name} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    ELSE()
    ENDIF()
    target_include_directories(${example_name}
      PRIVATE
      3rdparty
      3rdparty/imgui
      include
      ${INCLUDES}
      "$ENV{VULKAN_SDK}/include"
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(${example_name}
    ${Vulkan_LIBRARY}
    meshoptimizer
    ${LIBS}
    )

endfunction()

add_example(compute_rasterization examples/compute_rasterization.cpp)
add_example(draw_indirect examples/draw_indirect.cpp)
add_example(gpu_feedback examples/gpu_feedback.cpp)
add_example(simple_rendering examples/simple_rendering.cpp)
add_example(mesh_lod examples/mesh_lod.cpp)
add_example(simple_compute examples/simple_compute.cpp)
add_example(oit examples/oit.cpp)
add_example(virtual_textures examples/virtual_textures.cpp)
add_example(sdf examples/sdf.cpp)

add_executable(tests tests.cpp)
target_include_directories(tests
      PRIVATE
      3rdparty
      3rdparty/imgui
      include
      ${INCLUDES}
      "$ENV{VULKAN_SDK}/include"
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}
    )
