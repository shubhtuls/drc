CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
set(CMAKE_C_FLAGS "-std=c99")

FIND_PACKAGE(Torch REQUIRED)

# Flags
# When using MSVC
IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
ENDIF(MSVC)

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src init.c grid.c gridReg.c gridProj.c rayPotSem.c rayPotColor.c rayPotGeom.c rpUtils.c rayFusion.c mleDepthSem.c)
FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(rpsem "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(rpsem luaT TH)

