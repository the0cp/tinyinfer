#include <tinyinfer/compute/ops.h>
#include <tinyinfer/core/thread_pool.h>

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
    Tensor bt = random_tensor({512, 512});

    ThreadPool pool(8);
    bench("matmul_bt_serial", [&](){ return matmul_bt(a, bt); });
    bench("matmul_bt_8", [&](){ return matmul_bt(a, bt, &pool); });
    return 0;
}
