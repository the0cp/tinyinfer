#pragma once

#include <cstddef>
#include <memory>

namespace tinyinfer{

// A Buffer owns a stable, contiguous byte range. Implementations must keep the
// returned address valid and unchanged for the lifetime of the Buffer object.
class Buffer{
public:
    virtual ~Buffer() = default;

    virtual void* data() noexcept = 0;
    virtual const void* data() const noexcept = 0;
    virtual size_t size_bytes() const noexcept = 0;
};

class CpuBuffer final : public Buffer{
public:
    explicit CpuBuffer(size_t size_bytes);
    ~CpuBuffer() override;

    CpuBuffer(const CpuBuffer&) = delete;
    CpuBuffer& operator=(const CpuBuffer&) = delete;
    CpuBuffer(CpuBuffer&&) = delete;
    CpuBuffer& operator=(CpuBuffer&&) = delete;

    void* data() noexcept override;
    const void* data() const noexcept override;
    size_t size_bytes() const noexcept override;

private:
    void* data_ = nullptr;
    size_t size_bytes_ = 0;
};

struct CpuBufferStats{
    size_t live_bytes;
    size_t peak_bytes;
};

// Counts CpuBuffer payloads only; peak_bytes covers the process lifetime.
CpuBufferStats cpu_buffer_stats() noexcept;
std::shared_ptr<Buffer> make_cpu_buffer(size_t size_bytes);

}
