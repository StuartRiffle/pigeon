// tune.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_TUNE_H__
#define PIGEON_TUNE_H__


class AutoTuner : public AmoebaOptimizer
{
    struct TestPos
    {
        int     mCategory;
        float   mExpected;
        float   mWeight;
        int     mPad;

        u64     mWhitePieces[6];
        u64     mBlackPieces[6];

        bool operator<( const TestPos& rhs ) const { return( mCategory < rhs.mCategory ); }
    };

    enum
    {
        CATEGORIES = 3 * 3 * 3 * 2,
    };

    std::vector< TestPos >  mTestPos;
    Evaluator               mEvaluator;
    size_t                  mCategoryCount[CATEGORIES];
    size_t                  mCategoryOffset[CATEGORIES];
    int                     mCurrCategory;

public:
    AutoTuner()
    {
        mCurrCategory = 0;
        mTestPos.reserve( 100000 );
    }


    void PrepForCategory( int cat )
    {
        ParameterSet init(  384 + 6 );
        ParameterSet range( 384 + 6 );

        mCurrCategory = cat;

        double baseVals[] =
        {
           100.00000,   100.00000,   100.00000,   100.00000,   100.00000,   100.00000,   100.00000,   100.00000,
            72.76183,    96.09842,    90.38721,    68.98488,    62.31587,    58.46389,    77.11660,    73.50525,
            77.51245,    95.52872,    82.70342,    71.51050,    64.28857,    61.11927,    79.65089,    70.77631,
            75.70196,    92.62453,    84.54225,    83.62682,    82.65262,    70.73822,    83.50734,    80.82265,
            89.37450,   100.75786,    87.06840,    92.27415,    83.17789,    70.77252,    94.48904,    96.11889,
           116.90906,   133.14401,   123.47721,   122.34427,   129.61096,   133.22156,   138.72032,   132.11106,
           135.86698,   156.45297,   147.54337,   159.22866,   165.99136,   184.35623,   192.57754,   176.33213,
           100.00000,   100.00000,   100.00000,   100.00000,   100.00000,   100.00000,   100.00000,   100.00000,

           176.81977,   234.14587,   210.31067,   214.84655,   210.89025,   198.19970,   219.13176,   171.97918,
           219.39012,   219.59065,   235.39343,   235.03026,   235.43758,   227.01553,   216.15005,   190.63738,
           225.34987,   246.09573,   248.04851,   261.23302,   257.97976,   244.08118,   232.03594,   219.38066,
           247.77587,   273.25032,   271.32182,   269.31270,   259.33478,   265.81796,   255.43873,   232.66468,
           279.39880,   276.23638,   298.03382,   279.12702,   293.36223,   280.67897,   263.49128,   256.77229,
           282.35901,   289.91169,   308.75440,   300.19634,   298.26126,   282.92274,   276.46052,   252.83927,
           265.61951,   263.38853,   290.93821,   299.03562,   283.52784,   281.52835,   253.90068,   243.37726,
           167.69660,   263.03135,   263.99945,   277.40659,   262.99064,   252.09415,   250.81818,   145.24337,

           242.91970,   254.08989,   259.06581,   249.74579,   253.88544,   256.60480,   261.84068,   253.21924,
           259.34885,   275.06722,   271.06219,   268.58215,   260.56087,   275.51430,   265.90123,   268.61652,
           270.72480,   273.44239,   274.97005,   281.57843,   277.53875,   279.16636,   282.43066,   270.75332,
           270.11482,   280.92445,   275.86224,   291.63398,   295.82534,   285.12008,   283.92411,   278.44394,
           284.98338,   269.26482,   298.31691,   298.74708,   302.73347,   293.35790,   275.24417,   273.73940,
           298.87787,   302.40551,   304.41313,   302.51388,   303.23590,   293.68089,   295.18584,   281.06504,
           279.77345,   279.05711,   293.24613,   287.29364,   286.49266,   289.93929,   288.87644,   277.33590,
           264.53538,   277.14082,   270.49659,   280.51874,   278.86972,   276.42615,   275.69857,   270.20869,

           392.23487,   413.25649,   403.07265,   410.97156,   414.02107,   411.62721,   405.47904,   393.84309,
           398.30929,   409.53666,   405.13703,   405.12203,   406.04226,   407.26168,   402.64064,   394.32056,
           414.69477,   427.06326,   415.19802,   412.44829,   411.49882,   414.99296,   418.73876,   409.01812,
           427.77076,   438.23228,   435.84460,   431.17899,   433.15788,   433.45945,   434.91286,   427.30880,
           448.88731,   453.18538,   456.78107,   451.02763,   452.83986,   451.32274,   450.67830,   449.93251,
           460.44713,   467.67752,   469.21327,   466.14099,   465.82314,   465.50427,   463.13365,   458.16903,
           463.09040,   468.36595,   470.27981,   469.51119,   471.98993,   470.89068,   464.52834,   462.30742,
           464.16616,   458.42187,   459.52149,   461.10269,   463.27548,   460.84327,   459.03987,   456.40909,

           795.57359,   767.70740,   770.19707,   799.18590,   809.32253,   788.30785,   785.88369,   806.68335,
           798.22676,   790.12597,   799.03688,   812.05843,   812.30731,   804.77417,   805.98130,   805.20081,
           834.91374,   836.23076,   824.46550,   820.43513,   812.73654,   818.51685,   805.42313,   812.16087,
           851.78607,   853.98893,   845.58605,   840.20155,   829.65609,   828.46477,   826.38649,   809.34246,
           880.18709,   892.13832,   879.99258,   872.34110,   865.24730,   851.99874,   833.57278,   821.88710,
           908.84144,   946.82929,   926.62880,   900.73861,   878.97948,   862.47279,   856.77943,   831.21700,
           936.06866,   910.36390,   911.74205,   893.25304,   882.81149,   874.14388,   849.02859,   837.33712,
           923.79704,   905.55011,   902.31677,   896.08930,   883.83992,   872.55374,   864.33902,   853.94412,

           100+ -22.30863,   100+ -10.36884,   100+ -38.15179,   100+ -24.29733,   100+ -39.92837,   100+ -12.92425,   100+   4.31706,   100+  -1.05603,
           100+ -20.69664,   100+ -15.05791,   100+ -21.28083,   100+ -22.00771,   100+ -20.57522,   100+ -20.71243,   100+ -13.54479,   100+  -4.77419,
           100+ -26.90684,   100+ -14.25021,   100+ -12.91901,   100+ -10.60979,   100+ -12.37363,   100+ -11.82961,   100+  -9.68757,   100+ -19.88726,
           100+ -19.24021,   100+  -2.95137,   100+  -0.77311,   100+  -2.48195,   100+  -2.32299,   100+  -0.27138,   100+   1.76295,   100+ -13.17388,
           100+ -10.61768,   100+   2.91038,   100+   2.90915,   100+  -2.54789,   100+  -3.03725,   100+   7.70121,   100+   8.41085,   100+  -5.26058,
           100+  -6.39829,   100+  27.98944,   100+  13.69404,   100+   2.32638,   100+   6.16026,   100+  14.84491,   100+  23.43373,   100+   1.64037,
           100+  14.21232,   100+  38.87496,   100+  24.36778,   100+  16.37229,   100+  16.30848,   100+  21.07669,   100+  28.18398,   100+   3.88080,
           100+  -3.21499,   100+  34.70990,   100+  23.80101,   100+  15.95810,   100+  12.17889,   100+   8.05479,   100+  19.46922,   100+  -5.72823,
        };

        double initVals[] = 
        {
            0
        };

        init.mElem[0] = 100;
        init.mElem[1] = 300;
        init.mElem[2] = 300;
        init.mElem[3] = 500;
        init.mElem[4] = 900;
        init.mElem[5] = 100;  

#if 0
        FILE* fout = fopen( "weights.inc", "w" );

        for( int ccat = 0; ccat < CATEGORIES; ccat++ )
        {
            int i = ccat;
            int numKnights = i % 3; i /= 3;
            int numBishops = i % 3; i /= 3;
            int numRooks   = i % 3; i /= 3;
            int numQueens  = i;

            char catDesc[80] = { 0 };

            for( int j = 0; j < numQueens;  j++ ) strcat( catDesc, "Q" );
            for( int j = 0; j < numRooks;   j++ ) strcat( catDesc, "R" );
            for( int j = 0; j < numBishops; j++ ) strcat( catDesc, "B" );
            for( int j = 0; j < numKnights; j++ ) strcat( catDesc, "N" );

            fprintf( fout, "    // %s (%d positions)\n\n", catDesc, mCategoryCount[ccat] );

            int catOfs = ccat * 64 * 6;

            for( int piece = 0; piece < 6; piece++ )
            {
                int pieceOfs = catOfs + piece * 64;

                for( int y = 0; y < 8; y++ )
                {
                    int lineOfs = pieceOfs + y * 8;

                    std::string shortDesc = "";
                    std::string deltaDesc = "";

                    fprintf( fout, "    " );

                    for( int x = 0; x < 8; x++ )
                    {
                        double val = initVals[lineOfs + x];
                        if( val < 0 )
                            val = 0;

                        double delta = val - baseVals[piece * 64 + y * 8 + x];

                        if( ((piece == KNIGHT) && (numKnights == 0)) ||
                            ((piece == BISHOP) && (numBishops == 0)) ||
                            ((piece == ROOK)   && (numRooks   == 0)) ||
                            ((piece == QUEEN)  && (numQueens  == 0)) )
                        {
                            val = init.mElem[piece];
                            delta = 0;
                        }

                        char str[80];
                        sprintf( str, "%5d", (int) val );
                        shortDesc += str;

                        sprintf( str, "%4d", (int) delta );
                        deltaDesc += str;

                        int valFixed = (int) (val * WEIGHT_SCALE);

                        fprintf( fout, "0x%08x, ", valFixed );
                    }

                    fprintf( fout, " // %s   %s\n", shortDesc.c_str(), deltaDesc.c_str() );
                }

                fprintf( fout, "\n" );
            }
        }

        fclose( fout );
#endif

        bool piecesOfType[6] = { 0 };

        int i = cat;
        int numKnights = i % 3; i /= 3;
        int numBishops = i % 3; i /= 3;
        int numRooks   = i % 3; i /= 3;
        int numQueens  = i;
        
        piecesOfType[PAWN] = true;
        piecesOfType[KNIGHT] = (numKnights > 0);
        piecesOfType[BISHOP] = (numBishops > 0);
        piecesOfType[ROOK] = (numRooks > 0);
        piecesOfType[QUEEN] = (numQueens > 0);
        piecesOfType[KING] = true;

        init.mElem[0] = 100;
        init.mElem[1] = piecesOfType[KNIGHT]? 300 : 0;
        init.mElem[2] = piecesOfType[BISHOP]? 300 : 0;
        init.mElem[3] = piecesOfType[ROOK]?   500 : 0;
        init.mElem[4] = piecesOfType[QUEEN]?  900 : 0;
        init.mElem[5] = 100;  

        for( int i = 0; i < 384 + 6; i++ )
        {
            range.mElem[i] = (i < 6)? 0 : 1;//((i & 1)? 10 : -10);

            if( i >= 6 )
            {

                int initIdx = i - 6;
                int pieceType = initIdx / 64;
                double baseVal = init.mElem[initIdx / 64];
                int catBase = cat * 384;

                //init.mElem[i] = (initVals[catBase + initIdx] - baseVal);// * 10;
                init.mElem[i] = (baseVals[initIdx] - baseVal);// * 10;

                if( (pieceType != PAWN) && (pieceType != KING) )
                    init.mElem[i] = 0;

                if( !piecesOfType[pieceType] )
                {
                    range.mElem[i] = 0;
                    init.mElem[i] = 0;
                }
            }
            //range.mElem[i] = (i < 6)? 0 : ((init.mElem[i] > 0)? -10 : 10);
        }
        
        for( int i = 6; i < 6 + 8; i++ )
        {
            range.mElem[i] = 0;
            range.mElem[i + 7*8] = 0;
        }

        this->Initialize( init, range );
    }

    bool LoadGameLine( const char* str )
    {
        Tokenizer tok( str );

        float expected = 1;

        if( tok.Consume( "W" ) )
            expected = 1.0f;
        else if( tok.Consume( "L" ) )
            expected = 0.0f;
        else if( tok.Consume( "D" ) )
            expected = 0.5f;
        else
            return( false );

        Position pos;
        pos.Reset();

        for( const char* movetext = tok.ConsumeNext(); movetext; movetext = tok.ConsumeNext() )
        {
            MoveSpec spec;
            FEN::StringToMoveSpec( movetext, spec );
            if( spec.IsPromotion() )
                break;

            pos.Step( spec );
            expected = 1.0f - expected;

            MoveMap mmap;
            pos.CalcMoveMap( &mmap );

            if( this->IsValidTestPos( pos, mmap ) )
            {
                TestPos testPos = { 0 };

                testPos.mExpected = expected;

                float phase = mEvaluator.CalcGamePhase< 1 >( pos );
                //testPos.mWeight = 1.0f - abs( 1.0f - phase ); // MIDGAME
                //if( phase <= 1 )
                //    testPos.mWeight = 1.0f;

                //testPos.mWeight = 1;

                //if( (phase >= 1.0f) && (phase < 1.9f) )//testPos.mWeight > 0.3f )
                //if( phase < 1 ) // OPENING
                if( phase > 0.0f ) // ENDGAME
                //if( testPos.mWeight > 0 ) // MIDGAME
                {
                    //testPos.mWeight = phase;//phase - 1;
                    //testPos.mWeight = 2.0 - phase; // ENDGAME
                    //testPos.mWeight = 1.0f - phase; // OPENING    '
                    testPos.mWeight = 1.0f;

                    //u64 coeff[EVAL_TERMS];
                    //mEvaluator.CalcEvalTerms< 1, u64 >( pos, mmap, coeff );

                    testPos.mWhitePieces[PAWN]   = pos.mWhitePawns;
                    testPos.mWhitePieces[KNIGHT] = pos.mWhiteKnights;
                    testPos.mWhitePieces[BISHOP] = pos.mWhiteBishops;
                    testPos.mWhitePieces[ROOK]   = pos.mWhiteRooks;
                    testPos.mWhitePieces[QUEEN]  = pos.mWhiteQueens;
                    testPos.mWhitePieces[KING]   = pos.mWhiteKing;

                    testPos.mBlackPieces[PAWN]   = pos.mBlackPawns;
                    testPos.mBlackPieces[KNIGHT] = pos.mBlackKnights;
                    testPos.mBlackPieces[BISHOP] = pos.mBlackBishops;
                    testPos.mBlackPieces[ROOK]   = pos.mBlackRooks;
                    testPos.mBlackPieces[QUEEN]  = pos.mBlackQueens;
                    testPos.mBlackPieces[KING]   = pos.mBlackKing;

                    //for( int i = 0; i < EVAL_TERMS; i++ )
                    //    testPos.mCoeff[i] = (int) coeff[i];

                    int numPawns   = (int) CountBits< 1 >( pos.mWhitePawns   );
                    int numKnights = (int) CountBits< 1 >( pos.mWhiteKnights );
                    int numBishops = (int) CountBits< 1 >( pos.mWhiteBishops );
                    int numRooks   = (int) CountBits< 1 >( pos.mWhiteRooks   );
                    int numQueens  = (int) CountBits< 1 >( pos.mWhiteQueens  );

                    if( (numKnights <= 2) && (numBishops <= 2) && (numRooks <= 2) && (numQueens <= 1) )
                    {
                        //int cat = numPawns;
                        //cat *= 3;   cat += numKnights;
                        //cat *= 3;   cat += numBishops;
                        //cat *= 3;   cat += numRooks;  
                        //cat *= 2;   cat += numQueens; 

                        int cat = numQueens;
                        cat *= 3; cat += numRooks;
                        cat *= 3; cat += numBishops;
                        cat *= 3; cat += numKnights;

                        mCategoryCount[cat]++;

                        testPos.mCategory = cat;
                        mTestPos.push_back( testPos );
                    }

                }
            }
        }

        return( true );
    }

    void Serialize( const char* filename )
    {
        std::random_shuffle( mTestPos.begin(), mTestPos.end() );
        std::sort( mTestPos.begin(), mTestPos.end() );

        FILE* fout = fopen( filename, "wb" );
        size_t count = mTestPos.size();
        size_t written = 0;

        while( written < count )
        {
            written += fwrite( &mTestPos[written], sizeof( TestPos ), Min< u64 >( count - written, 1024 * 1024 ), fout );
        }

        fclose( fout );
    }

    void IndexCategories()
    {
        for( size_t i = 0; i < CATEGORIES; i++ )
        {
            mCategoryCount[i] = 0;
            mCategoryOffset[i] = 0;
        }

        for( size_t i = 0; i < mTestPos.size(); i++ )
        {
            int cat = mTestPos[i].mCategory;

            mCategoryCount[cat]++;

            if( (i > 0) && (mCategoryOffset[cat] == 0) )
                mCategoryOffset[cat] = i;
        }
    }

    void Deserialize( const char* filename )
    {
#if PIGEON_MSVC
        // Yuck- I had to switch this to the native Windows API for large files to work properly.

        HANDLE fin = CreateFileA( filename, GENERIC_READ, 0, NULL, OPEN_ALWAYS, 0, NULL );
        DWORD llo, lhi;
        llo = GetFileSize( fin, &lhi );

        u64 len = ((u64) lhi) << 32ULL | llo;
        size_t count = len / sizeof( TestPos );
        size_t numRead = 0;

        mTestPos.resize( count );

        static TestPos* checkArray = &mTestPos[0];

        while( numRead < count )
        {
            size_t maxBytes = 1024 * 1024 * 1024;
            size_t batchSize = maxBytes / sizeof( TestPos ); 
            size_t remaining = count - numRead;

            if( batchSize > remaining )
                batchSize = remaining;

            if( !ReadFile( fin, &mTestPos[numRead], (DWORD) (batchSize * sizeof( TestPos )), NULL, NULL ) )
                break;

            numRead += batchSize;
        }

        assert( numRead == count );
        CloseHandle( fin );

        this->IndexCategories();

        //std::random_shuffle( mTestPos.begin(), mTestPos.end() );
#else
        fprintf( stderr, "ERROR: Deserialize() not implemented!\n" );
#endif
    }

    void Dump( bool logToFile = false )
    {
        const ParameterSet& best = this->GetBest();

        printf( "\n\n" );
        printf( "%" PRId64 " iterations, cat %d using %d test cases, error %.15f\n", this->GetIterCount(), mCurrCategory, (int) mCategoryCount[mCurrCategory], best.mError );

        if( logToFile )
        {
            char filename[80];
            sprintf( filename, "cat%02d.txt", mCurrCategory );

            FILE* flog = fopen( filename, "w" );
            for( int pieces = 0; pieces < 6; pieces++ )
            {
                double pieceBase = best.mElem[pieces];

                for( int i = 0; i < 64; i++ )
                {
                    if( (i > 0) && (i % 8 == 0) )
                        fprintf( flog, "\n" );

                    int idx = 6 + (pieces * 64) + i;
                    fprintf( flog, "%12.5f,", pieceBase + best.mElem[idx] );
                }
                fprintf( flog, "\n\n" );
            }

            fprintf( flog, "\n" );
            fclose( flog );
        }
    }

protected:
    bool IsValidTestPos( const Position& pos, const MoveMap& mmap )
    {
        MoveList moves;
        moves.UnpackMoveMap( pos, mmap );
        moves.DiscardMovesBelow( CAPTURE_LOSING );

        // Only accept quiet positions

        if( moves.mCount == 0 )
            return( true );

        return( false );
    }

    virtual void CalcError( ParameterSet& point )
    {
        double  accum   = 0;
        double  divisor = 0;
        double  factor  = 0.7 / 100;



        size_t catOffset = mCategoryOffset[mCurrCategory];
        size_t numIters  = mCategoryCount[mCurrCategory];

        //catOffset = 0;
        //numIters = mTestPos.size();

        #pragma omp parallel for reduction( +: accum, divisor ) schedule( static )
        for( i64 i = 0; i < (i64) numIters; i++ )
        {
            const TestPos& testPos = mTestPos[catOffset + i];

            double score = 0;

            for( int pieces = 0; pieces < 6; pieces++ )
            {
                double pieceBase = point.mElem[pieces];
                int offset = 6 + (pieces * 64);

                u64 bits = testPos.mWhitePieces[pieces];
                while( bits )
                {
                    u64 idx = ConsumeLowestBitIndex( bits );
                    score += (pieceBase + point.mElem[offset + idx]);
                }

                bits = ByteSwap( testPos.mBlackPieces[pieces] );
                while( bits )
                {
                    u64 idx = ConsumeLowestBitIndex( bits );
                    score -= (pieceBase + point.mElem[offset + idx]);
                }
            }

            double sig = 1.0 / (1.0 + exp( -score * factor ));
            double err = (sig - testPos.mExpected);

            //err *= testPos.mWeight;

            accum   = accum   + testPos.mWeight * (err * err);
            divisor = divisor + testPos.mWeight;
        }

        point.mError = (accum / divisor);
    }
};


#endif // PIGEON_TUNE_H__
};

