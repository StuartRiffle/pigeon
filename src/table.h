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
    u8          mProbe;


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
            ((u64) mFailHigh    <<  2) |
            ((u64) mProbe       <<  0) );
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
        mProbe      = (u8)   ((val >>  0) & 3);
    }
};

struct TableBucket
{
    enum
    {
        SHIFT = 3,
        SLOTS = (1 << SHIFT),
        MASK  = (SLOTS - 1)
    };

    u64*        mSource;
    int         mIdxFound;
    TableEntry  mEntry[SLOTS];

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

    INLINE PDECL bool IsBetterEntry( const TableEntry& curr, const TableEntry& prev ) const
    {
        if( curr.mDepth > prev.mDepth )
            return( true );

        bool currExact = !(curr.mFailHigh || curr.mFailLow);
        bool prevExact = !(prev.mFailHigh || prev.mFailLow);

        if( currExact && !prevExact )
            return( true );

        return( false );
    }

    PDECL void LoadBucket( const u64& hash, TableBucket& bucket ) const
    {
        uintptr_t   bucketIndex = (hash & mMask) >> TableBucket::SHIFT;
        u64*        bucketSlots = mTable + (bucketIndex << TableBucket::SHIFT);
        u32         hashVerify  = (u32) (hash >> 40);

        bucket.mSource = bucketSlots;
        bucket.mIdxFound  = -1;

        for( int i = 0; i < TableBucket::SLOTS; i++ )
        {
            TableEntry* entry = bucket.mEntry + i;

            entry->Unpack( bucketSlots[i] );
            if( entry->mHashVerify == hashVerify )
                bucket.mIdxFound = i;
        }

        //printf( "%c", (bucket.mIdxFound >= 0)? 'f' : ('A' + poss) );
    }

    PDECL void StoreBucket( const TableBucket& bucket, const TableEntry& tt )
    {
        int prevIdx = bucket.mIdxFound;
        if( prevIdx >= 0 )
        {
            const TableEntry& prev = bucket.mEntry[prevIdx];

            if( !this->IsBetterEntry( prev, tt ) )
                bucket.mSource[prevIdx] = tt.Pack();
        }
        else
        {
            int worst = 0;

            for( int i = 1; i < TableBucket::SLOTS; i++ )
                if( this->IsBetterEntry( bucket.mEntry[worst], bucket.mEntry[i] ) )
                    worst = i;

            bucket.mSource[worst] = tt.Pack();
        }
    }

};


#endif // PIGEON_TABLE_H__
};
