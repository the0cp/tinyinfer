#include "ops.h"
#include "thread_pool.h"

#include <chrono>
#include <iostream>
#include <random>
#include <string>

using namespace tinyinfer;

static Tensor random_tensor(Shape shape){
    Tensor tensor(std::move(shape));
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    for(size_t i = 0; i < tensor.numel(); i++){
        tensor.data()[i] = distribution(generator);
    }

    return tensor;
}

template<typename Function>
static void bench(const std::string& name, Function&& function){
    const auto start = std::chrono::steady_clock::now();
    Tensor result = function();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );
    std::cout << name << ": " << elapsed.count()
              << " ms, check=" << result.data()[0] << "\n";
}

int main(){
    Tensor a = random_tensor({512, 512});
    Tensor b = random_tensor({512, 512});

    bench("fast_matmul", [&](){ return fast_matmul(a, b); });
    bench("blocked_matmul_32", [&](){ return blocked_matmul(a, b, 32); });
    bench("parallel_matmul_8", [&](){ return parallel_matmul(a, b, 8); });

    ThreadPool pool(8);
    bench("threadpool_matmul_8", [&](){ return threadpool_matmul(a, b, pool, 8); });
    return 0;
}
