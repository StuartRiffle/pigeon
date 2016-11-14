// platform.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle

#ifndef PIGEON_PLATFORM_H__
#define PIGEON_PLATFORM_H__

// This file has gotten a bit hairy, and could do with some cleanup!

#ifndef _HAS_EXCEPTIONS    
#define _HAS_EXCEPTIONS   0
#endif

#include <stdint.h>
#include <assert.h>

#if PIGEON_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#if PIGEON_ENABLE_CUDA && defined( __CUDA_ARCH__ )

    // We are running __device__ code

    #define PIGEON_CUDA_DEVICE  (1)
    #define PIGEON_ALLOW_POPCNT (1)
    #define PIGEON_ALIGN( _N )  __align__( _N )
    #define PIGEON_ALIGN_SIMD   __align__( 32 )    

    #define RESTRICT            __restrict
    #define INLINE              __forceinline__    
    #define PDECL               __device__

#elif defined( _MSC_VER )

    #define WIN32_LEAN_AND_MEAN    
    #include <windows.h>
    #include <process.h>
    #include <intrin.h>
    #include <limits.h>

    #include <atomic>

    #pragma warning( disable: 4996 )    // CRT security warnings
    #pragma warning( disable: 4293 )    // Shift count negative or too big (due to unused branch in templated function)
    #pragma warning( disable: 4752 )    // Found Intel(R) Advanced Vector Extensions; consider using /arch:AVX
    #pragma warning( disable: 4554 )    // Check operator precedence for possible error; use parentheses to clarify precedence (false warning caused by nvcc compile)
    #pragma inline_recursion( on )
    #pragma inline_depth( 255 )
    
    #define PIGEON_CPU          (1)
    #define PIGEON_MSVC         (1)
    #define PIGEON_ENABLE_SSE2  (1)
    #define PIGEON_ENABLE_SSE4  (1)
    #define PIGEON_ENABLE_AVX2  (1)
    #define PIGEON_ALLOW_POPCNT (1)
    #define PIGEON_ALIGN( _N )  __declspec( align( _N ) )
    #define PIGEON_ALIGN_SIMD   __declspec( align( 32 ) )

    #define RESTRICT            __restrict
    #define DEBUGBREAK          __debugbreak
    #define INLINE              __forceinline
    #define PDECL         
    #define PRId64              "I64d"

    extern "C" void * __cdecl memset(void *, int, size_t);
    #pragma intrinsic( memset )        

#elif defined( __GNUC__ )

    #define __STDC_FORMAT_MACROS

    #include <inttypes.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <emmintrin.h>
    #include <cpuid.h>
    #include <string.h>
    #include <unistd.h>

    #include <atomic>

    #pragma GCC diagnostic ignored "-Wunknown-pragmas"

    #define PIGEON_CPU          (1)
    #define PIGEON_GCC          (1)
    #define PIGEON_ALLOW_POPCNT (1)
    #define PIGEON_ALIGN( _N )  __attribute__(( aligned( _N ) ))
    #define PIGEON_ALIGN_SIMD   __attribute__(( aligned( 32 ) ))    

    #define RESTRICT            __restrict
    #define DEBUGBREAK          void
    #define INLINE              inline __attribute__(( always_inline ))
    #define PDECL         

    #define stricmp             strcasecmp
    #define strnicmp            strncasecmp

#else
    #error
#endif

#if PIGEON_ENABLE_CUDA && PIGEON_CPU
#define PIGEON_CUDA_HOST (1)
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

    enum
    {
        CPU_X64,
        CPU_SSE2,
        CPU_SSE4,
        CPU_AVX2,
        CPU_AVX3,

        CPU_LEVELS
    };

    INLINE PDECL u64 PlatByteSwap64( const u64& val )             
    { 
    #if PIGEON_CUDA_DEVICE
        u32 hi = __byte_perm( (u32) val, 0, 0x0123 );
        u32 lo = __byte_perm( (u32) (val >> 32), 0, 0x0123 );
        return( ((u64) hi << 32ULL) | lo );
    #elif PIGEON_MSVC
        return( _byteswap_uint64( val ) ); 
    #elif PIGEON_GCC
        return( __builtin_bswap64( val ) );     
    #endif
    }

    INLINE PDECL u64 PlatLowestBitIndex64( const u64& val )
    {
    #if PIGEON_CUDA_DEVICE
         return( __ffsll( val ) - 1 );
    #elif PIGEON_MSVC
        unsigned long result;
        _BitScanForward64( &result, val );
        return( result );
    #elif PIGEON_GCC
        return( __builtin_ffsll( val ) - 1 ); 
    #endif
    }

    INLINE PDECL u64 SoftCountBits64( const u64& val )
    {
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

    template< int POPCNT >
    INLINE PDECL u64 PlatCountBits64( const u64& val )
    {
    #if PIGEON_CUDA_DEVICE
        return( __popcll( val ) );
    #else
        #if PIGEON_ALLOW_POPCNT
            #if PIGEON_MSVC
                if( POPCNT ) 
                    return( __popcnt64( val ) );
                else // SoftCountBits64
            #elif PIGEON_GCC
                if( POPCNT ) 
                    return( __builtin_popcountll( val ) );
                else // SoftCountBits64
            #endif
        #endif

        return( SoftCountBits64( val ) );
    #endif
    }

    INLINE PDECL void PlatClearMemory( void* mem, size_t bytes )
    {
    #if PIGEON_CUDA_DEVICE
        memset( mem, 0, bytes );
    #elif PIGEON_MSVC
        ::memset( mem, 0, bytes );
    #elif PIGEON_GCC
        __builtin_memset( mem, 0, bytes );    
    #endif
    }

    INLINE PDECL void PlatPrefetch( void* mem )
    {
    #if PIGEON_MSVC
        _mm_prefetch( (char*) mem, _MM_HINT_NTA );
    #elif PIGEON_GCC
        __builtin_prefetch( mem );  
    #endif
    }

    INLINE PDECL void PlatMemoryFence()
    {
    #if PIGEON_CUDA_DEVICE
        __threadfence();
    #else
        std::atomic_thread_fence( std::memory_order_seq_cst );
    #endif
    }

#if !PIGEON_CUDA_DEVICE

    INLINE PDECL bool PlatCheckCpuFlag( int leaf, int idxWord, int idxBit )
    {
    #if PIGEON_MSVC
        int info[4] = { 0 };
        __cpuid( info, leaf );
    #elif PIGEON_GCC
        unsigned int info[4] = { 0 };
        if( !__get_cpuid( leaf, info + 0, info + 1, info + 2, info + 3 ) )
            return( false );
    #endif

        return( (info[idxWord] & (1 << idxBit)) != 0 );
    }

    INLINE PDECL bool PlatDetectPopcnt()
    {
    #if PIGEON_MSVC
        return( PlatCheckCpuFlag( 1, 2, 23 ) );
    #elif PIGEON_GCC
        return( __builtin_cpu_supports( "popcnt" ) );
    #endif
    }

    INLINE PDECL int PlatDetectCpuLevel()
    {
    #if PIGEON_ENABLE_AVX2
        if( PlatCheckCpuFlag( 7, 1, 5 ) )   return( CPU_AVX2 );
    #endif
    #if PIGEON_ENABLE_SSE4
        if( PlatCheckCpuFlag( 1, 2, 20 ) )  return( CPU_SSE4 );
    #endif
    #if PIGEON_ENABLE_SSE2
        if( PlatCheckCpuFlag( 1, 3, 26 ) )  return( CPU_SSE2 );
    #endif

        return( CPU_X64 );
    }

    INLINE PDECL int PlatDetectCpuCores()
    {
    #if PIGEON_MSVC
        SYSTEM_INFO si = { 0 };
        GetSystemInfo( &si );
        return( si.dwNumberOfProcessors );
    #elif PIGEON_GCC
        return( sysconf( _SC_NPROCESSORS_ONLN ) );
    #endif
    }

    INLINE PDECL ThreadId PlatSpawnThread( void* (*func)( void* ), void* arg )
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


    INLINE PDECL void PlatSetThreadName( const char* name )
    {
    #if PIGEON_MSVC
        //#pragma pack( push, 8 )
        struct THREADNAME_INFO
        {
            DWORD   dwType;     
            LPCSTR  szName;     
            DWORD   dwThreadID; 
            DWORD   dwFlags;    
        };
        //#pragma pack( pop )

        THREADNAME_INFO info;

        info.dwType     = 0x1000;
        info.szName     = name;
        info.dwThreadID = GetCurrentThreadId();
        info.dwFlags    = 0;

        __try
        {
            const DWORD MS_VC_EXCEPTION = 0x406D1388;
            RaiseException( MS_VC_EXCEPTION, 0, sizeof( info ) / sizeof( ULONG_PTR ), (ULONG_PTR*) &info );
        }
        __except( EXCEPTION_EXECUTE_HANDLER ) {}
    #endif
    }


    INLINE PDECL void PlatSleep( int ms )
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

    struct Mutex
    {
    #if PIGEON_MSVC
        CRITICAL_SECTION mCritSec;
        Mutex()      { InitializeCriticalSection( &mCritSec ); }
       ~Mutex()      { DeleteCriticalSection( &mCritSec ); }
        void Enter() { EnterCriticalSection( &mCritSec ); }
        void Leave() { LeaveCriticalSection( &mCritSec ); }
    #elif PIGEON_GCC
        pthread_mutexattr_t mAttrib;
        pthread_mutex_t     mMutex;
        Mutex()      { pthread_mutexattr_init( &mAttrib ); pthread_mutexattr_settype( &mAttrib, PTHREAD_MUTEX_RECURSIVE ); pthread_mutex_init( &mMutex, &mAttrib ); }
       ~Mutex()      { pthread_mutex_destroy( &mMutex ); pthread_mutexattr_destroy( &mAttrib ); }
        void Enter() { pthread_mutex_lock( &mMutex ); }
        void Leave() { pthread_mutex_unlock( &mMutex ); }
    #endif

        class Scope
        {
            Mutex& mMutex;
        public:
            Scope( Mutex& mutex ) : mMutex( mutex ) { mMutex.Enter(); }
           ~Scope() { mMutex.Leave(); }
        };
    };

    struct Timer
    {
        u64     mStartTime;

        Timer() { this->Reset(); }
        Timer( const Timer& rhs ) : mStartTime( rhs.mStartTime ) {}

        void    Reset()         { mStartTime = Timer::GetTick(); }
        i64     GetElapsedMs()  { return( ((i64) (Timer::GetTick() - mStartTime) * 1000) / Timer::GetFrequency() ); }

        static u64 GetTick()
        { 
    #if PIGEON_MSVC
            LARGE_INTEGER tick; 
            QueryPerformanceCounter( &tick ); 
            return( tick.QuadPart ); 
    #elif PIGEON_GCC
            return( (u64) clock() );
    #endif
        }

        static u64 GetFrequency()
        {
    #if PIGEON_MSVC
            static LARGE_INTEGER freq = { 0 };
            if( !freq.QuadPart )
                QueryPerformanceFrequency( &freq );
            return( freq.QuadPart );
    #elif PIGEON_GCC
            return( (u64) CLOCKS_PER_SEC );
    #endif
        }
    };

#endif




};

#endif // PIGEON_PLATFORM_H__
