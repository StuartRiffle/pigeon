project( gaviota-tb )
cmake_minimum_required( VERSION 3.2 )

add_definitions( -DNDEBUG )
add_definitions( -DZ_PREFIX )
add_definitions( -D_CRT_SECURE_NO_WARNINGS )

include_directories( "." )
include_directories( "sysport" )
include_directories( "compression" )
include_directories( "compression/huffman" )
include_directories( "compression/liblzf" )
include_directories( "compression/zlib" )
include_directories( "compression/lzma" )

#if( ${CMAKE_CXX_COMPILER_ID} MATCHES Clang )
#
#	set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" )
#	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )
#	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -v -static-libgcc -Bstatic" )
#	set( CMAKE_STATIC_LINKER_FLAGS, "${CMAKE_STATIC_LINKER_FLAGS} -pie -v" )
#
#endif()

add_library( ${PROJECT_NAME} STATIC
    gtb-probe.c
    gtb-dec.c
    gtb-att.c
    sysport/sysport.c  
    compression/wrap.c
    compression/huffman/hzip.c 
    compression/liblzf/lzf_c.c
    compression/liblzf/lzf_d.c
    compression/zlib/zcompress.c
    compression/zlib/uncompr.c
    compression/zlib/inflate.c
    compression/zlib/deflate.c
    compression/zlib/adler32.c   
    compression/zlib/crc32.c    
    compression/zlib/infback.c  
    compression/zlib/inffast.c  
    compression/zlib/inftrees.c  
    compression/zlib/trees.c     
    compression/zlib/zutil.c
    compression/lzma/LzmaEnc.c 
    compression/lzma/LzmaDec.c 
    compression/lzma/Alloc.c 
    compression/lzma/LzFind.c 
    compression/lzma/Lzma86Enc.c 
    compression/lzma/Lzma86Dec.c 
    compression/lzma/Bra86.c
)


#if( ${CMAKE_CXX_COMPILER_ID} MATCHES Clang )
#
#	set_property( TARGET gaviota-tb PROPERTY POSITION_INDEPENDENT_CODE ON )
#	target_compile_options( gaviota-tb PRIVATE -fPIC )
#
#endif()

