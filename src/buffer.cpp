#include "buffer.h"

#include <new>

namespace tinyinfer{

CpuBuffer::CpuBuffer(size_t size_bytes) : size_bytes_(size_bytes){
    if(size_bytes_ != 0){
        data_ = ::operator new(size_bytes_);
    }
}

CpuBuffer::~CpuBuffer(){
    ::operator delete(data_);
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

std::shared_ptr<Buffer> make_cpu_buffer(size_t size_bytes){
    return std::make_shared<CpuBuffer>(size_bytes);
}

}
