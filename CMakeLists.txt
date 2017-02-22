
project( pigeon )

cmake_minimum_required( VERSION 2.8 )

if( ${CMAKE_CXX_COMPILER_ID} MATCHES MSVC )

	add_compile_options( /GS- /GL /W3 /Gy- /Zc:wchar_t /Zi /Gm- /Ox /Ob2 /fp:fast /fp:except- /WX- /GR- /Gd /Oy /Oi /MT )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG" )

elseif( (${CMAKE_CXX_COMPILER_ID} MATCHES GNU) OR (${CMAKE_CXX_COMPILER_ID} MATCHES Clang) )

	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -pthread -w -std=c++11" )

endif()

add_subdirectory( contrib/gaviota-tb-0.4.5 )
include_directories( contrib )

add_executable( pigeon src/pigeon.cpp )
target_link_libraries( pigeon gaviota-tb )


### find_package( CUDA QUIET REQUIRED )
### set( CUDA_NVCC_FLAGS ${} -Xcudafe "--diag_suppress=negative_shift_count" -cudart static -use_fast_math -DPIGEON_ENABLE_CUDA=1 )
### cuda_add_executable( pigeon-cuda src/pigeon.cpp src/kernel.cu )
