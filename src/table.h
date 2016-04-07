// table.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_TABLE_H__
#define PIGEON_TABLE_H__


/// Hash table entry for a Position that has been evaluated
//
struct TableEntry
{
    u32         mHashVerify;    
    i16         mScore;
    u8          mDepth;
    u8          mBestSrc;
    u8          mBestDest;
    bool        mFailLow;
    bool        mFailHigh;
    bool        mWhiteMove;

    INLINE u64 Pack() const
    {
        return(
            ((u64) mHashVerify  << 40) |
            ((u64) (u16) mScore << 24) |
            ((u64) mDepth       << 16) |
            ((u64) mBestSrc     << 10) |
            ((u64) mBestDest    <<  4) |
            ((u64) mFailLow     <<  3) |
            ((u64) mFailHigh    <<  2) |
            ((u64) mWhiteMove   <<  1) );
    }

    INLINE void Unpack( const u64& val )
    {
        mHashVerify = (u32)   (val >> 40);
        mScore      = (i16)   (val >> 24);
        mDepth      = (u8)    (val >> 16);
        mBestSrc    = (u8)   ((val >> 10) & 0x3F);
        mBestDest   = (u8)   ((val >>  4) & 0x3F);
        mFailLow    = (bool) ((val >>  3) & 1);
        mFailHigh   = (bool) ((val >>  2) & 1);
        mWhiteMove  = (bool) ((val >>  1) & 1);
    }
};


class HashTable
{
    u64*            mTable;
    u64             mMask;
    int             mEntries;

public:
    HashTable() :
        mTable( NULL ),
        mMask( 0 ),
        mEntries( 0 ) {}

    ~HashTable()
    {
        if( mTable )
            delete[] mTable;
    }

    void SetSize( size_t megs )
    {
        if( mTable )
            delete[] mTable;

        size_t bytes = megs * 1024 * 1024;

        u64 keyBits = 1;
        while( ((1ULL << (keyBits + 1)) * sizeof( u64 )) <= bytes )
            keyBits++;

        mEntries = 1 << keyBits;
        mMask    = mEntries - 1;
        mTable   = new u64[mEntries];

        this->Clear();
    }

    int GetSize() const
    {
        return( mEntries * sizeof( u64 ) / (1024 * 1024) ); 
    }

    void Clear()
    {
        PlatClearMemory( mTable, mEntries * sizeof( u64 ) );
    }

    float CalcUtilization() const
    {
        i64 totalUsed = 0;

        for( int i = 0; i < mEntries; i++ )
            if( mTable[i] )
                totalUsed++;

        float utilization = totalUsed * 1.0f / mEntries;
        return( utilization );
    }

    INLINE void Prefetch( const u64& hash ) const
    {
        PlatPrefetch( mTable + (hash & mMask) );
    }

    INLINE void Load( const u64& hash, TableEntry& tt ) const
    {
        tt.Unpack( mTable[hash & mMask] );
    }

    INLINE void Store( const u64& hash, const TableEntry& tt )
    {
        mTable[hash & mMask] = tt.Pack();
    }
};


#endif // PIGEON_TABLE_H__
};
