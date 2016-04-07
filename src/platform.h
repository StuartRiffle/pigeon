// platform.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_PLATFORM_H__
#define PIGEON_PLATFORM_H__

#include <stdint.h>

#if defined( _MSC_VER )

    #define WIN32_LEAN_AND_MEAN    
    #include <windows.h>
    #include <process.h>
    #include <intrin.h>
    #include <limits.h>

    #pragma warning( disable: 4996 ) // CRT security warnings
    #pragma warning( disable: 4293 ) // Shift count negative or too big (due to unused branch in templated function)
    #pragma inline_recursion( on )
    #pragma inline_depth( 255 )
    
    #define PIGEON_MSVC         (1)
    #define PIGEON_ENABLE_SSE2  (1)
    #define PIGEON_ENABLE_SSE4  (1)
    #define PIGEON_ENABLE_AVX2  (1)
    #define RESTRICT            __restrict
    #define DEBUGBREAK          __debugbreak
    #define INLINE              __forceinline
    #define PRId64              "I64d"
    #define _HAS_EXCEPTIONS     (0)
    #define PIGEON_ALLOW_POPCNT (1)

    
    extern "C" void * __cdecl memset(void *, int, size_t);
    #pragma intrinsic( memset )        

#elif defined( __GNUC__ )

    #define __STDC_FORMAT_MACROS

    #include <inttypes.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <emmintrin.h>
    #include <string.h>


    #pragma GCC diagnostic ignored "-Wunknown-pragmas"

    #define PIGEON_GCC          (1)
    #define PIGEON_ENABLE_SSE2  (1)
    #define RESTRICT            __restrict
    #define DEBUGBREAK          void
    #define INLINE              inline __attribute__(( always_inline ))
    #define stricmp             strcasecmp
    #define strnicmp            strncasecmp

#else
    #error
#endif

namespace Pigeon 
{
    typedef uint64_t    u64;
    typedef  int64_t    i64;
    typedef uint32_t    u32;
    typedef  int32_t    i32;
    typedef uint16_t    u16;
    typedef  int16_t    i16;
    typedef  uint8_t    u8;
    typedef   int8_t    i8;

#if PIGEON_MSVC
    typedef uintptr_t   ThreadId;
#elif PIGEON_GCC
    typedef pthread_t   ThreadId;
#endif

    INLINE u64 PlatByteSwap64( const u64& val )             
    { 
    #if PIGEON_MSVC
        return( _byteswap_uint64( val ) ); 
    #elif PIGEON_GCC
        return( __builtin_bswap64( val ) );     
    #endif
    }

    INLINE u64 PlatLowestBitIndex64( const u64& val )
    {
    #if PIGEON_MSVC
        unsigned long result;
        _BitScanForward64( &result, val );
        return( result );
    #elif PIGEON_GCC
        return( __builtin_ffsll( val ) ); 
    #endif
    }

    INLINE bool PlatDetectPopcnt()
    {
    #if PIGEON_MSVC
        int info[4] = { 0 };
        __cpuid( info, 1 );
        return( (info[2] & (1 << 23)) != 0 );
    #elif PIGEON_GCC
        return( __builtin_cpu_supports( "popcnt" ) );
    #endif
    }

    template< int POPCNT >
    INLINE u64 PlatCountBits64( const u64& val )
    {
    #if PIGEON_ALLOW_POPCNT
        #if PIGEON_MSVC
            if( POPCNT ) return( __popcnt64( val ) );
        #elif PIGEON_GCC
            if( POPCNT ) return( __builtin_popcountll( val ) );
        #endif
    #endif

        const u64 mask01 = 0x0101010101010101ULL;
        const u64 mask0F = 0x0F0F0F0F0F0F0F0FULL;
        const u64 mask33 = 0x3333333333333333ULL;
        const u64 mask55 = 0x5555555555555555ULL;

        register u64 n = val;

        n =  n - ((n >> 1) & mask55);
        n = (n & mask33) + ((n >> 2) & mask33);
        n = (n + (n >> 4)) & mask0F;
        n = (n * mask01) >> 56;

        return( n );
    }

    INLINE void PlatClearMemory( void* mem, size_t bytes )
    {
    #if PIGEON_MSVC
        ::memset( mem, 0, bytes );
    #elif PIGEON_GCC
        __builtin_memset( mem, 0, bytes );    
    #endif
    }

    INLINE void PlatPrefetch( void* mem )
    {
    #if PIGEON_MSVC
        _mm_prefetch( (char*) mem, _MM_HINT_NTA );
    #elif PIGEON_GCC
        __builtin_prefetch( mem );  
    #endif
    }

    INLINE ThreadId PlatSpawnThread( void* (*func)( void* ), void* arg )
    {
    #if PIGEON_MSVC
        ThreadId id = _beginthread( reinterpret_cast< void (*)( void* ) >( func ), 0, arg ); 
        return( id );
    #elif PIGEON_GCC
        ThreadId id;
        pthread_create( &id, NULL, func, arg );
        return( id );
    #endif
    }

    INLINE void PlatSleep( int ms )
    {
    #if PIGEON_MSVC
        Sleep( ms );
    #elif PIGEON_GCC
        timespec request;
        timespec remaining;
        request.tv_sec  = (ms / 1000);
        request.tv_nsec = (ms % 1000) * 1000 * 1000;
        nanosleep( &request, &remaining );
    #endif
    }

    struct Semaphore
    {
    #if PIGEON_MSVC
        HANDLE      mHandle;
        Semaphore() : mHandle( CreateSemaphore( NULL, 0, LONG_MAX, NULL ) ) {}
       ~Semaphore() { CloseHandle( mHandle ); }
        void Post() { ReleaseSemaphore( mHandle, 1, NULL ); }
        void Wait() { WaitForSingleObject( mHandle, INFINITE ); }
    #elif PIGEON_GCC
        sem_t       mHandle;
        Semaphore() { sem_init( &mHandle, 0, 0 ); }
       ~Semaphore() { sem_destroy( &mHandle ); }
        void Post() { sem_post( &mHandle ); }
        void Wait() { while( sem_wait( &mHandle ) ) {} }
    #endif
    };


};

#endif // PIGEON_PLATFORM_H__
