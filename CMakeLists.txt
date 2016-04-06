cmake_minimum_required( VERSION 3.0 )

project( pigeon )
set( PIGEON_VERSION 1.32 )

if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
	add_compile_options( /GS- /GL /W3 /Gy- /Zc:wchar_t /Zi /Gm- /Ox /Ob2 /fp:fast /fp:except- /WX- /GR- /Gd /Oy /Oi /MT )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG" )
elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -pthread -w" )
endif()

add_executable( pigeon-${PIGEON_VERSION}-x64 src/pigeon.cpp )

