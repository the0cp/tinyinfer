#include "ops.h"
#include "thread_pool.h"

#include <chrono>
#include <iostream>
#include <random>
#include <string>

using namespace tinyinfer;

static Tensor random_tensor(std::vector<size_t>shape){
    Tensor t(shape);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for(size_t i = 0; i < t.numel(); i++){
        t.data()[i] = dist(rng);
    }

    return t;
}

template<typename Func>
static Tensor bench(const std::string& name, Func func){
    auto start = std::chrono::high_resolution_clock::now();

    Tensor c = func();

    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();

    std::cout << name << " took " << ms << " ms" << ", check = " << c.at({0, 0}) << "\n";

    return c;
}

int main() {
    const size_t m = 512;
    const size_t k = 512;
    const size_t n = 512;

    Tensor a = random_tensor({m, k});
    Tensor b = random_tensor({k, n});

    bench("naive_matmul", [&](){return naive_matmul(a, b);});
    bench("fast_matmul", [&](){return fast_matmul(a, b);});

    Tensor bt = transpose_2d(b);

    bench("matmul_transposed_b", [&](){return matmul_transposed_b(a, bt);});

    bench("blocked_matmul_16", [&](){return blocked_matmul(a, b, 16);});

    bench("blocked_matmul_32", [&](){return blocked_matmul(a, b, 32);});

    bench("blocked_matmul_64", [&](){return blocked_matmul(a, b, 64);});

    for(size_t threads : {2, 4, 8, 16}){
        Tensor c = bench("parallel_matmul_" + std::to_string(threads), 
            [&](){return parallel_matmul(a, b, threads);});
    }

    ThreadPool pool(8);
    Tensor c = bench("threadpool_matmul_8", [&]() {
        return threadpool_matmul(a, b, pool, 8);
    });

    return 0;
}