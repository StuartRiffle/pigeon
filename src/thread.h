// thread.h - PIGEON CHESS ENGINE (c) 2012-2016 Stuart Riffle
        
namespace Pigeon {
#ifndef PIGEON_THREAD_H__
#define PIGEON_THREAD_H__


template< typename T >
class ThreadSafeQueue
{
    Mutex           mMutex;
    Semaphore       mAvail;
    std::queue< T > mQueue;

public:
    void Push( const T& obj )
    {
        mMutex.Enter();
        mQueue.push( obj );
        mMutex.Leave();

        mAvail.Post();
    }   

    T Pop()
    {
        mAvail.Wait();

        mMutex.Enter();
        T result = mQueue.front();
        mQueue.pop();
        mMutex.Leave();

        return( result );
    }

    bool TryPop( T& result )
    {
        Mutex::Scope lock( mMutex );

        if( mQueue.empty() )
            return( false );

        result = mQueue.front();
        mQueue.pop();

        mAvail.Wait();
        return( true );
    }

    void Clear()
    {
        Mutex::Scope lock( mMutex );

        while( !mQueue.empty() )
        {
            mQueue.pop();
            mAvail.Wait();
        }
    }
};


class WorkerThread
{
    typedef std::function< void() >         Action;
    typedef std::shared_ptr< Action >       ActionRef;
    typedef ThreadSafeQueue< ActionRef >    ActionQueue;

    ActionQueue     mQueue;
    Semaphore       mSemaRunning;
    Semaphore       mSemaTerminated;
    std::string     mThreadName;
    volatile bool   mAborting;

public:
    WorkerThread( const char* name ) : 
        mThreadName( name ),
        mAborting( false )
    {
        PlatSpawnThread( &WorkerThread::ThreadProc, this );
        mSemaRunning.Wait();
    }

    ~WorkerThread()
    {
        mSemaTerminated.Wait();
    }

    template< typename T >
    void Enqueue( T& closure )
    {
        mQueue.Push( ActionRef( new Action( closure ) ) );
    }

    void FinishAndStop()
    {
        mQueue.Push( NULL );
    }

    void Abort()
    {
        mAborting = true;
        mQueue.Push( NULL );
    }

private:
    void Run()
    {
        PlatSetThreadName( mThreadName.c_str() );
        mSemaRunning.Post();

        for( ;; )
        {
            ActionRef actionRef = mQueue.Pop();
            if( !actionRef )
                break;

            if( mAborting )
            {
                mQueue.Clear();
                break;
            }

            Action& action = *actionRef;
            action();
        }
    }

    static void* ThreadProc( void* param )
    {
        WorkerThread* inst = (WorkerThread*) param;

        inst->Run();
        inst->mSemaTerminated.Post();

        return( NULL );
    }
};

#endif // PIGEON_THREAD_H__
};
