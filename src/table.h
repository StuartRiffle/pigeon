// table.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_TABLE_H__
#define PIGEON_TABLE_H__


/// Hash table entry for a Position that has been evaluated

struct TableEntry
{
    u32         mHashVerify;    
    i16         mScore;
    u8          mDepth;
    u8          mBestSrc;
    u8          mBestDest;
    u8          mFailLow;
    u8          mFailHigh;


    /// Pack all fields into a 64-bit word

    INLINE PDECL u64 Pack() const
    {
        return(
            ((u64) mHashVerify  << 40) |
            ((u64) (u16) mScore << 24) |
            ((u64) mDepth       << 16) |
            ((u64) mBestSrc     << 10) |
            ((u64) mBestDest    <<  4) |
            ((u64) mFailLow     <<  3) |
            ((u64) mFailHigh    <<  2) );
    }


    /// Unpack fields from a 64-bit word

    INLINE PDECL void Unpack( const u64& val )
    {
        mHashVerify = (u32)   (val >> 40);
        mScore      = (i16)   (val >> 24);
        mDepth      = (u8)    (val >> 16);
        mBestSrc    = (u8)   ((val >> 10) & 0x3F);
        mBestDest   = (u8)   ((val >>  4) & 0x3F);
        mFailLow    = (u8)   ((val >>  3) & 1);
        mFailHigh   = (u8)   ((val >>  2) & 1);
    }
};

template< int SLOTS = 8 >
struct TableBucket
{
    u64*        mSource;
    TableEntry  mEntry[SLOTS];

    TableEntry* Load( u64 hash, u64* src )
    {
        mSource = src;

        TableEntry* found = NULL;
        u32 hashVerify = (u32) (hash >> 40);

        for( int i = 0; i < SLOTS; i++ )
        {
            TableEntry* dest = mEntry + i;

            dest->Unpack( mSource[i] );

            if( dest->mHashVerify == hashVerify )
                found = dest;
        }

        return( found );
    }


};


/// Transposition table

struct HashTable
{
    u64*            mTable;
    u64             mMask;
    u64             mEntries;

    HashTable() :
        mTable( NULL ),
        mMask( 0 ),
        mEntries( 0 ) {}

    PDECL ~HashTable()
    {
#if !PIGEON_CUDA_DEVICE
        if( mTable )
            delete[] mTable;
#endif
    }

    void SetSize( size_t megs )
    {
        if( mTable )
            delete[] mTable;

        size_t bytes = megs * 1024 * 1024;
        this->CalcTableEntries( bytes );

        mTable = new u64[mEntries];

        this->Clear();
    }

    size_t GetSize() const
    {
        return( mEntries * sizeof( u64 ) / (1024 * 1024) ); 
    }

    void CalcTableEntries( size_t bytes )
    {
        u64 keyBits = 1;
        while( ((1ULL << (keyBits + 1)) * sizeof( u64 )) <= bytes )
            keyBits++;

        mEntries = 1ULL << keyBits;
        mMask    = mEntries - 1;
    }

    void Clear()
    {
        PlatClearMemory( mTable, mEntries * sizeof( u64 ) );
    }

    float EstimateUtilization() const
    {
        i64  totalUsed = 0;

        for( int i = 0; i < TT_SAMPLE_SIZE; i++ )
            if( mTable[i] )
                totalUsed++;

        float utilization = totalUsed * 1.0f / TT_SAMPLE_SIZE;
        return( utilization );
    }

    INLINE void Prefetch( const u64& hash ) const
    {
        PlatPrefetch( mTable + (hash & mMask) );
    }

    INLINE PDECL void Load( const u64& hash, TableEntry& tt ) const
    {
        u64* ptr = mTable + (hash & mMask);

#if PIGEON_CUDA_DEVICE
        u64 packed = atomicAdd( ptr, 0ULL );    // 64-bit atomic load (is there a better way to do this?)
#else
        u64 packed = *ptr;
#endif

        tt.Unpack( packed );
    }

    INLINE PDECL void Store( const u64& hash, const TableEntry& tt )
    {
        u64* ptr    = mTable + (hash & mMask);  
        u64  packed = tt.Pack();

#if PIGEON_CUDA_DEVICE
        atomicExch( ptr, packed );              // 64-bit atomic store (is there a better way to do this?)
#else
        *ptr = packed;
#endif
    }
};


#endif // PIGEON_TABLE_H__
};
