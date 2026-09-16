#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <iostream>
#include <thread>
#include <vector>
#include <cstring>  // for memcpy
#include <immintrin.h>
#include <sched.h>
#include <mutex>
#include <queue>
#include <utility>
#include <atomic>
#include <cassert>
#include <condition_variable>


void set_affinity(uint32_t idx) {
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(idx, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

// Thread pool
// Note that this thread pool should be reused across multiple layers
class MyThreadPool {
public:
    void Start(uint32_t num_threads = 0);
    void QueueJobWOLock(const std::function<void(void*)>& job, void* para);
    void NotifyAll();
    void NotifyOne();
    void Stop();
    void Wait();
    void AddNumTask(int);
    void DisplayNumTask();
    void LockQueue();
    void UnlockQueue();

private:
    void ThreadLoop(uint32_t); // input is the thread id

    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::mutex main_mutex;
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
    std::condition_variable main_condition; // main thread uses this condition variable to wait
    std::vector<std::thread> threads;
    std::queue<std::pair<std::function<void(void*)>,void*>> jobs;
    std::atomic<int> num_tasks;
};

void MyThreadPool::Start(uint32_t num_threads) {
    if(num_threads == 0){
        num_threads = std::thread::hardware_concurrency();
    }

    for (uint32_t ii = 0; ii < num_threads; ++ii) {
        threads.emplace_back(std::thread(&MyThreadPool::ThreadLoop, this, ii));
    }

    num_tasks = 0;
}

void MyThreadPool::ThreadLoop(uint32_t thread_idx) {
    set_affinity(thread_idx); 
    while (true) {
        std::pair<std::function<void(void*)>,void*> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !jobs.empty() || should_terminate;
            });
            if (should_terminate) {
                return;
            }
            job = jobs.front();
            jobs.pop();

        }
        job.first(job.second);
        auto cur_num = num_tasks.fetch_sub(1);
        if(cur_num == 1){
            // std::cout << "finally notify main thread" << std::endl;
            std::unique_lock<std::mutex> lock(main_mutex);
            main_condition.notify_one();
        }
    }
}

void MyThreadPool::QueueJobWOLock(const std::function<void(void*)>& job, void* para) {
    jobs.push(std::pair<std::function<void(void*)>,void*>(job, para));
}

void MyThreadPool::AddNumTask(int num){
    num_tasks.fetch_add(num);
    // assert(num_tasks == jobs.size());
}

void MyThreadPool::DisplayNumTask(){
    std::cout << "Num tasks = " << num_tasks << std::endl;
}

void MyThreadPool::NotifyAll() {
    mutex_condition.notify_all();
}

void MyThreadPool::NotifyOne() {
    mutex_condition.notify_one();
}

void MyThreadPool::LockQueue() {
    queue_mutex.lock();
}

void MyThreadPool::UnlockQueue() {
    queue_mutex.unlock();
}

// Only use this when the system terminates
void MyThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread& active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
}

// Wait until all submitted tasks have been executed
void MyThreadPool::Wait(){
    {
        std::unique_lock<std::mutex> lock(main_mutex);
        main_condition.wait(lock, [this] {
            return num_tasks == 0;
        });
    }
}

#endif // THREAD_POOL_HPP