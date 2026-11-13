# Builds only the import stubs of adrenaline_kernel (libAdrenalineKernel_stub.a)
# that adrenaline_user links against. The stubs depend only on kernel/exports.yml,
# so a placeholder implementation is enough; the module itself is not rebuilt
# (CI ships the prebuilt bubble/pkg/sce_module/adrenaline_kernel.skprx).
cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Please define VITASDK to point to your SDK path!")
  endif()
endif()

project(adrenaline_kernel_stubs)
include("${VITASDK}/share/vita.cmake" REQUIRED)

set(EXPORTS ${CMAKE_SOURCE_DIR}/../../kernel/exports.yml)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-q -Wall -O2 -nostdlib")

add_executable(adrenaline_kernel stubs.c)

vita_create_stubs(stubs adrenaline_kernel ${EXPORTS} KERNEL)

install(DIRECTORY ${CMAKE_BINARY_DIR}/stubs/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.a"
)
