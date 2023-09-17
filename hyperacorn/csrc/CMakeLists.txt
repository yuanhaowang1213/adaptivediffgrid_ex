


set(TARGET_NAME "hyperacorn")



add_library(LIB_GEOBASIC SHARED IMPORTED)
set_target_properties(LIB_GEOBASIC PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/geometry/libgeobasic.so)


add_library(LIB_DATABASIC SHARED IMPORTED)
set_target_properties(LIB_DATABASIC PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/data/libdatabasic.so)
#set(lib_geobasic_path ${CMAKE_CURRENT_SOURCE_DIR}/geometry/)

#find_library(geobasic libgeobasic)


## Generate source file with git commit
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC MY_GIT_SHA1 ALLOW_LOOKING_ABOVE_CMAKE_SOURCE_DIR)
set(MY_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")


#configure_file (
#        "${CMAKE_CURRENT_SOURCE_DIR}/git_sha1.h.in"
#        "${MY_BUILDCONFIG_DIR}/git_sha1.h"
#        @ONLY
#)

#configure_file (
#        "${CMAKE_CURRENT_SOURCE_DIR}/build_config.h.in"
#        "${MY_BUILDCONFIG_DIR}/build_config.h"
#        @ONLY
#)
message("libs is ")
message("${LIBS}")

add_library(${TARGET_NAME} SHARED "")
target_include_directories(${TARGET_NAME} PUBLIC "." ${MY_BUILDCONFIG_DIR})
#target_sources(${TARGET_NAME} PRIVATE "${MY_BUILDCONFIG_DIR}/build_config.h" "${MY_BUILDCONFIG_DIR}/git_sha1.h")
target_sources(${TARGET_NAME} PRIVATE "")

#message(FATAL_ERROR ${LIBS})
target_link_libraries(${TARGET_NAME} ${LIBS} LIB_GEOBASIC LIB_DATABASIC -lstdc++fs)
#target_link_libraries(${TARGET_NAME} PRIVATE ${geobasic})

#target_link_libraries(nr TorchVision::TorchVision)

list(APPEND MY_CUDA_FLAGS "--expt-relaxed-constexpr")
list(APPEND MY_CUDA_FLAGS "-lineinfo")

target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_FLAGS}>)
set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 70-virtual)
#############  source #############


# Source files in "src/" to level directory
FILE(GLOB_RECURSE PROG_SRC  *.cpp)
FILE(GLOB_RECURSE PROG_CUDA_SRC  *.cu)
FILE(GLOB_RECURSE PROG_HEADER  *.h ${SAIGA_BUILDCONFIG_FILE})
target_sources(${TARGET_NAME} PRIVATE ${PROG_SRC} ${PROG_CUDA_SRC} PUBLIC ${PROG_HEADER})


set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

#add_dependencies(${TARGET_NAME} pytorch)


if (PR_NO_WINDOW)
    target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
endif ()
