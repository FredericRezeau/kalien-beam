/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>
*/

#pragma once
#include "ports/sim.cuh"

__device__ __forceinline__ float fitness(const Simulation& sim, int32_t wave) {
    float f = (float)sim.score;
    if ((sim.gameOver && sim.lives <= 0) || (wave > 0 && sim.wave > wave)) {
        f *= 0.01f; // Heavy penalty for failing
    } else if (wave > 0 && sim.wave == wave && sim.astCount == 1) {
        // Avoid cold fields to maximize our aggressive ship behavior.
        f += (float)sim.saucerCount * 1000.0f;
        if (!sim.ship.canControl) {
            f -= 2000.0f;
        }
        float r = (float)sim.score / (float)sim.frameCount;
        if (r < 39.04f) {
            f -= (39.04f - r) * 500.0f; // Penalize below theoretical rate.
        }
    } else {
        f += (float)sim.wave * 500.0f;
        f += (float)(ASTEROID_CAP - sim.astCount) * 50.0f;
        f -= (float)sim.frameCount * 100.0f;
    }
    return f;
}
