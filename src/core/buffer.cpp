#include <tinyinfer/core/buffer.h>

#include <atomic>
#include <new>

namespace tinyinfer{

namespace{

constinit std::atomic_size_t live_bytes{0};
constinit std::atomic_size_t peak_bytes{0};

void record_allocation(size_t bytes) noexcept{
    const size_t live = live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    size_t peak = peak_bytes.load(std::memory_order_relaxed);

    while(
        live > peak &&
        !peak_bytes.compare_exchange_weak(
            peak,
            live,
            std::memory_order_relaxed,
            std::memory_order_relaxed
        )
    ){}
}

}

CpuBuffer::CpuBuffer(size_t size_bytes) : size_bytes_(size_bytes){
    if(size_bytes_ != 0){
        data_ = ::operator new(size_bytes_);
        record_allocation(size_bytes_);
    }
}

CpuBuffer::~CpuBuffer(){
    ::operator delete(data_);
    live_bytes.fetch_sub(size_bytes_, std::memory_order_relaxed);
}

void* CpuBuffer::data() noexcept{
    return data_;
}

const void* CpuBuffer::data() const noexcept{
    return data_;
}

size_t CpuBuffer::size_bytes() const noexcept{
    return size_bytes_;
}

CpuBufferStats cpu_buffer_stats() noexcept{
    const size_t live = live_bytes.load(std::memory_order_relaxed);
    const size_t peak = peak_bytes.load(std::memory_order_relaxed);
    return {live, peak < live ? live : peak};
}

std::shared_ptr<Buffer> make_cpu_buffer(size_t size_bytes){
    return std::make_shared<CpuBuffer>(size_bytes);
}

}
