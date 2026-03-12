/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>
*/

#pragma once
#include <vector>

#ifndef __CUDACC__
#include <algorithm>

void prune(const float* fitness, int total, int K, std::vector<int>& top) {
    top.resize(total);
    for (int i = 0; i < total; i++) {
        top[i] = i;
    }
    std::nth_element(top.begin(), top.begin() + K, top.end(),
        [fitness](int a, int b) { return fitness[a] > fitness[b]; });
    top.resize(K);
}
#else
#include "fitness.cuh"
void prune(const float* fitness, int total, int K, std::vector<int>& top);
#endif