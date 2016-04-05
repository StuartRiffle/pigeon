cmake_minimum_required( VERSION 3.3 )

project( pigeon )
set( PIGEON_VERSION 1.30 )

if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" )
    add_compile_options( "-std=c++11" )
elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
    add_compile_options( "" )
endif()

add_executable( pigeon-${PIGEON_VERSION}-uci src/pigeon.cpp )

