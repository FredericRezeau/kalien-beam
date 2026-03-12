/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>

    Bit-perfect C++/CUDA port of https://github.com/kalepail/kalien for ZK verification.

    IMPORTANT: This code must maintain **bit-exact behavior** for the ZK verification pipeline.
    Do not modify unless you understand the deterministic guarantees required for ZK and/or
    maintenance parity is required for original implementation updates.
*/

#pragma once
#ifndef __NVRTC__
#include <cfloat>
#include <cstdint>
#include <cstring>
#endif

#ifndef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __constant__
#define __constant__
#endif
#include <algorithm>
#include <cstdint>
using std::abs;
using std::max;
using std::min;
#endif

static constexpr int32_t WORLD_WIDTH = 960;
static constexpr int32_t WORLD_HEIGHT = 720;
static constexpr int32_t STARTING_LIVES = 3;
static constexpr int32_t EXTRA_LIFE_SCORE_STEP = 10000;
static constexpr int32_t SHIP_RADIUS = 14;
static constexpr int32_t SHIP_TURN_SPEED_BAM = 3;
static constexpr int32_t SHIP_FACING_UP_BAM = 192;
static constexpr int32_t SHIP_THRUST_Q8_8 = 20;
static constexpr int32_t SHIP_MAX_SPEED_SQ_Q16_16 = 1451 * 1451;
static constexpr int32_t SHIP_BULLET_SPEED_Q8_8 = 2219;
static constexpr int32_t SHIP_BULLET_LIMIT = 4;
static constexpr int32_t SHIP_BULLET_LIFETIME_FRAMES = 72;
static constexpr int32_t SHIP_BULLET_COOLDOWN_FRAMES = 10;
static constexpr int32_t SHIP_RESPAWN_FRAMES = 75;
static constexpr int32_t SHIP_SPAWN_INVULNERABLE_FRAMES = 120;
static constexpr int32_t SAUCER_BULLET_SPEED_Q8_8 = 1195;
static constexpr int32_t SAUCER_BULLET_LIMIT = 2;
static constexpr int32_t SAUCER_BULLET_LIFETIME_FRAMES = 72;
static constexpr int32_t SAUCER_SPAWN_MIN_FRAMES = 420;
static constexpr int32_t SAUCER_SPAWN_MAX_FRAMES = 840;
static constexpr int32_t ASTEROID_CAP = 27;
static constexpr int32_t SCORE_LARGE_ASTEROID = 20;
static constexpr int32_t SCORE_MEDIUM_ASTEROID = 50;
static constexpr int32_t SCORE_SMALL_ASTEROID = 100;
static constexpr int32_t SCORE_LARGE_SAUCER = 200;
static constexpr int32_t SCORE_SMALL_SAUCER = 990;
static constexpr int32_t WORLD_WIDTH_Q12_4 = 15360;
static constexpr int32_t WORLD_HEIGHT_Q12_4 = 11520;
static constexpr int32_t AST_SPEED_LARGE_MIN = 145;
static constexpr int32_t AST_SPEED_LARGE_MAX = 248;
static constexpr int32_t AST_SPEED_MEDIUM_MIN = 265;
static constexpr int32_t AST_SPEED_MEDIUM_MAX = 401;
static constexpr int32_t AST_SPEED_SMALL_MIN = 418;
static constexpr int32_t AST_SPEED_SMALL_MAX = 606;
static constexpr int32_t SAUCER_SPEED_SMALL_Q8_8 = 405;
static constexpr int32_t SAUCER_SPEED_LARGE_Q8_8 = 299;
static constexpr int32_t MAX_GAME_FRAMES = 36000;
static constexpr int32_t LURK_TIME_THRESHOLD_FRAMES = 360;
static constexpr int32_t LURK_SAUCER_SPAWN_FAST_FRAMES = 180;
static constexpr int32_t AST_RADIUS_LARGE = 48;
static constexpr int32_t AST_RADIUS_MEDIUM = 28;
static constexpr int32_t AST_RADIUS_SMALL = 16;
static constexpr int32_t SAUCER_RADIUS_LARGE = 22;
static constexpr int32_t SAUCER_RADIUS_SMALL = 16;
static constexpr int32_t SHIP_RESPAWN_EDGE_PADDING_Q12_4 = 1536;
static constexpr int32_t SHIP_RESPAWN_GRID_STEP_Q12_4 = 1024;
static constexpr int32_t SAUCER_START_X_LEFT_Q12_4 = -480;
static constexpr int32_t SAUCER_START_X_RIGHT_Q12_4 = 15840;
static constexpr int32_t SAUCER_START_Y_MIN_Q12_4 = 1152;
static constexpr int32_t SAUCER_START_Y_MAX_Q12_4 = 10368;
static constexpr int32_t SAUCER_CULL_MIN_X_Q12_4 = -1280;
static constexpr int32_t SAUCER_CULL_MAX_X_Q12_4 = 16640;
static constexpr int64_t WAVE_SAFE_DIST_SQ_Q24_8 = (int64_t)2880 * 2880;
static constexpr int32_t SHIP_BSPEED_Q88 = SHIP_BULLET_SPEED_Q8_8;
static constexpr int32_t SHIP_BLIMIT = SHIP_BULLET_LIMIT;
static constexpr int32_t SAUCER_BLIMIT = SAUCER_BULLET_LIMIT;
static constexpr uint8_t INPUT_LEFT = 0x1;
static constexpr uint8_t INPUT_RIGHT = 0x2;
static constexpr uint8_t INPUT_THRUST = 0x4;
static constexpr uint8_t INPUT_FIRE = 0x8;
static constexpr int32_t AST_LARGE = 0;
static constexpr int32_t AST_MEDIUM = 1;
static constexpr int32_t AST_SMALL = 2;

__device__ __constant__ int16_t SIM_SIN_TABLE[256] = {
    0, 402, 804, 1205, 1606, 2006, 2404, 2801, 3196, 3590, 3981, 4370, 4756, 5139, 5520, 5897,
    6270, 6639, 7005, 7366, 7723, 8076, 8423, 8765, 9102, 9434, 9760, 10080, 10394, 10702, 11003, 11297,
    11585, 11866, 12140, 12406, 12665, 12916, 13160, 13395, 13623, 13842, 14053, 14256, 14449, 14635,
    14811, 14978, 15137, 15286, 15426, 15557, 15679, 15791, 15893, 15986, 16069, 16143, 16207, 16261,
    16305, 16340, 16364, 16379, 16384, 16379, 16364, 16340, 16305, 16261, 16207, 16143, 16069, 15986,
    15893, 15791, 15679, 15557, 15426, 15286, 15137, 14978, 14811, 14635, 14449, 14256, 14053, 13842,
    13623, 13395, 13160, 12916, 12665, 12406, 12140, 11866, 11585, 11297, 11003, 10702, 10394, 10080,
    9760, 9434, 9102, 8765, 8423, 8076, 7723, 7366, 7005, 6639, 6270, 5897, 5520, 5139, 4756, 4370,
    3981, 3590, 3196, 2801, 2404, 2006, 1606, 1205, 804, 402, 0, -402, -804, -1205, -1606, -2006,
    -2404, -2801, -3196, -3590, -3981, -4370, -4756, -5139, -5520, -5897, -6270, -6639, -7005, -7366,
    -7723, -8076, -8423, -8765, -9102, -9434, -9760, -10080, -10394, -10702, -11003, -11297, -11585,
    -11866, -12140, -12406, -12665, -12916, -13160, -13395, -13623, -13842, -14053, -14256, -14449,
    -14635, -14811, -14978, -15137, -15286, -15426, -15557, -15679, -15791, -15893, -15986, -16069,
    -16143, -16207, -16261, -16305, -16340, -16364, -16379, -16384, -16379, -16364, -16340, -16305,
    -16261, -16207, -16143, -16069, -15986, -15893, -15791, -15679, -15557, -15426, -15286, -15137,
    -14978, -14811, -14635, -14449, -14256, -14053, -13842, -13623, -13395, -13160, -12916, -12665,
    -12406, -12140, -11866, -11585, -11297, -11003, -10702, -10394, -10080, -9760, -9434, -9102,
    -8765, -8423, -8076, -7723, -7366, -7005, -6639, -6270, -5897, -5520, -5139, -4756, -4370, -3981,
    -3590, -3196, -2801, -2404, -2006, -1606, -1205, -804, -402};

__device__ __constant__ int16_t SIM_COS_TABLE[256] = {
    16384, 16379, 16364, 16340, 16305, 16261, 16207, 16143, 16069, 15986, 15893, 15791, 15679, 15557,
    15426, 15286, 15137, 14978, 14811, 14635, 14449, 14256, 14053, 13842, 13623, 13395, 13160, 12916,
    12665, 12406, 12140, 11866, 11585, 11297, 11003, 10702, 10394, 10080, 9760, 9434, 9102, 8765,
    8423, 8076, 7723, 7366, 7005, 6639, 6270, 5897, 5520, 5139, 4756, 4370, 3981, 3590, 3196, 2801,
    2404, 2006, 1606, 1205, 804, 402, 0, -402, -804, -1205, -1606, -2006, -2404, -2801, -3196, -3590,
    -3981, -4370, -4756, -5139, -5520, -5897, -6270, -6639, -7005, -7366, -7723, -8076, -8423, -8765,
    -9102, -9434, -9760, -10080, -10394, -10702, -11003, -11297, -11585, -11866, -12140, -12406,
    -12665, -12916, -13160, -13395, -13623, -13842, -14053, -14256, -14449, -14635, -14811, -14978,
    -15137, -15286, -15426, -15557, -15679, -15791, -15893, -15986, -16069, -16143, -16207, -16261,
    -16305, -16340, -16364, -16379, -16384, -16379, -16364, -16340, -16305, -16261, -16207, -16143,
    -16069, -15986, -15893, -15791, -15679, -15557, -15426, -15286, -15137, -14978, -14811, -14635,
    -14449, -14256, -14053, -13842, -13623, -13395, -13160, -12916, -12665, -12406, -12140, -11866,
    -11585, -11297, -11003, -10702, -10394, -10080, -9760, -9434, -9102, -8765, -8423, -8076, -7723,
    -7366, -7005, -6639, -6270, -5897, -5520, -5139, -4756, -4370, -3981, -3590, -3196, -2801, -2404,
    -2006, -1606, -1205, -804, -402, 0, 402, 804, 1205, 1606, 2006, 2404, 2801, 3196, 3590, 3981,
    4370, 4756, 5139, 5520, 5897, 6270, 6639, 7005, 7366, 7723, 8076, 8423, 8765, 9102, 9434, 9760,
    10080, 10394, 10702, 11003, 11297, 11585, 11866, 12140, 12406, 12665, 12916, 13160, 13395, 13623,
    13842, 14053, 14256, 14449, 14635, 14811, 14978, 15137, 15286, 15426, 15557, 15679, 15791, 15893,
    15986, 16069, 16143, 16207, 16261, 16305, 16340, 16364, 16379};

__device__ __constant__ uint8_t SIM_ATAN_TABLE[33] = {
    0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25, 26, 27,
    28, 29, 29, 30, 31, 31, 32};

__device__ __forceinline__ int32_t simSin(int32_t angle) {
    return SIM_SIN_TABLE[angle & 0xff];
}

__device__ __forceinline__ int32_t simCos(int32_t angle) {
    return SIM_COS_TABLE[angle & 0xff];
}

__device__ __forceinline__ int32_t simAtan2(int32_t dy, int32_t dx) {
    if (dx == 0 && dy == 0) {
        return 0;
    }

    int32_t absDx = dx < 0 ? -dx : dx;
    int32_t absDy = dy < 0 ? -dy : dy;

    int32_t ratio;
    bool swapped;
    if (absDx >= absDy) {
        ratio = (absDx == 0) ? 0 : (int32_t)(((int64_t)absDy << 5) / absDx);
        swapped = false;
    } else {
        ratio = (absDy == 0) ? 0 : (int32_t)(((int64_t)absDx << 5) / absDy);
        swapped = true;
    }

    if (ratio > 32) {
        ratio = 32;
    }

    int32_t angle = SIM_ATAN_TABLE[ratio];

    if (swapped) {
        angle = 64 - angle;
    }
    if (dx < 0) {
        angle = 128 - angle;
    }
    if (dy < 0) {
        angle = (256 - angle) & 0xff;
    }

    return angle & 0xff;
}

__device__ __forceinline__ void simDisplace(
    int32_t angle, int32_t distPixels, int32_t& dx, int32_t& dy) {
    dx = (simCos(angle) * distPixels) >> 10;
    dy = (simSin(angle) * distPixels) >> 10;
}

__device__ __forceinline__ void simVelocity(
    int32_t angle, int32_t speedQ8_8, int32_t& vx, int32_t& vy) {
    vx = (simCos(angle) * speedQ8_8) >> 14;
    vy = (simSin(angle) * speedQ8_8) >> 14;
}

__device__ __forceinline__ int32_t simApplyDrag(int32_t v) {
    return v - (v >> 7);
}

__device__ __forceinline__ void simClampSpeed(int32_t& vx, int32_t& vy, int32_t maxSqQ16_16) {
    int64_t speedSq = (int64_t)vx * vx + (int64_t)vy * vy;
    if (speedSq <= maxSqQ16_16) {
        return;
    }
    while (speedSq > maxSqQ16_16) {
        vx = (vx * 3) >> 2;
        vy = (vy * 3) >> 2;
        speedSq = (int64_t)vx * vx + (int64_t)vy * vy;
    }
}

__device__ __forceinline__ int32_t simWrapX(int32_t x) {
    if ((uint32_t)x < (uint32_t)WORLD_WIDTH_Q12_4) {
        return x;
    }
    if (x < 0) {
        return x + WORLD_WIDTH_Q12_4;
    }
    return x - WORLD_WIDTH_Q12_4;
}

__device__ __forceinline__ int32_t simWrapY(int32_t y) {
    if ((uint32_t)y < (uint32_t)WORLD_HEIGHT_Q12_4) {
        return y;
    }
    if (y < 0) {
        return y + WORLD_HEIGHT_Q12_4;
    }
    return y - WORLD_HEIGHT_Q12_4;
}

__device__ __forceinline__ int32_t shortDeltaQ12_4(int32_t from, int32_t to, int32_t size) {
    int32_t delta = to - from;
    int32_t half = size >> 1;
    if (delta > half) {
        delta -= size;
    } else if (delta < -half) {
        delta += size;
    }
    return delta;
}

__device__ __forceinline__ int32_t shortDX(int32_t fromX, int32_t toX) {
    return shortDeltaQ12_4(fromX, toX, WORLD_WIDTH_Q12_4);
}

__device__ __forceinline__ int32_t shortDY(int32_t fromY, int32_t toY) {
    return shortDeltaQ12_4(fromY, toY, WORLD_HEIGHT_Q12_4);
}

struct Rng {
    uint32_t state;

    __device__ void seed(uint32_t s) {
        state = (s == 0) ? 0xdeadbeef : s;
    }

    __device__ uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return state;
    }

    __device__ int32_t nextRange(int32_t mn, int32_t mx) {
        return mn + (int32_t)(next() % (uint32_t)(mx - mn));
    }
};

__device__ __forceinline__ int64_t collisionDistSqQ12_4(
    int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    int32_t dx = shortDX(ax, bx);
    int32_t dy = shortDY(ay, by);
    return (int64_t)dx * dx + (int64_t)dy * dy;
}

__device__ __forceinline__ bool collidesQ12_4(
    int32_t ax, int32_t ay, int32_t ar,
    int32_t bx, int32_t by, int32_t br) {
    int32_t hitDist = (ar + br) << 4;
    int32_t negHitDist = -hitDist;
    int32_t dx = shortDX(ax, bx);
    if (dx < negHitDist || dx > hitDist) {
        return false;
    }
    int32_t dy = shortDY(ay, by);
    if (dy < negHitDist || dy > hitDist) {
        return false;
    }
    return (int64_t)dx * dx + (int64_t)dy * dy <= (int64_t)hitDist * hitDist;
}

__device__ __forceinline__ int64_t clearanceSqQ12_4(
    int32_t hazardX, int32_t hazardY, int32_t hazardRadius,
    int32_t spawnX, int32_t spawnY, int32_t spawnRadius) {
    int32_t hitDist = (hazardRadius + spawnRadius) << 4;
    int32_t dx = shortDX(hazardX, spawnX);
    int32_t dy = shortDY(hazardY, spawnY);
    return (int64_t)dx * dx + (int64_t)dy * dy - (int64_t)hitDist * hitDist;
}

__device__ __forceinline__ int32_t waveLargeAsteroidCount(int32_t wave) {
    if (wave <= 4) {
        return 4 + (wave - 1) * 2;
    }
    int32_t v = 10 + (wave - 4);
    return (v < 16) ? v : 16;
}

__device__ __forceinline__ int32_t maxSaucersForWave(int32_t wave) {
    if (wave < 4) {
        return 1;
    }
    if (wave < 7) {
        return 2;
    }
    return 3;
}

__device__ __forceinline__ int32_t astRadius(int32_t size) {
    if (size == AST_LARGE) {
        return AST_RADIUS_LARGE;
    }
    if (size == AST_MEDIUM) {
        return AST_RADIUS_MEDIUM;
    }
    return AST_RADIUS_SMALL;
}

__device__ __forceinline__ void astSpeedRange(int32_t size, int32_t& mn, int32_t& mx) {
    if (size == AST_LARGE) {
        mn = AST_SPEED_LARGE_MIN;
        mx = AST_SPEED_LARGE_MAX;
        return;
    }
    if (size == AST_MEDIUM) {
        mn = AST_SPEED_MEDIUM_MIN;
        mx = AST_SPEED_MEDIUM_MAX;
        return;
    }
    mn = AST_SPEED_SMALL_MIN;
    mx = AST_SPEED_SMALL_MAX;
}

struct Asteroid {
    int32_t x, y, vx, vy;
    int32_t angle;
    int32_t spin;
    int32_t radius;
    int32_t size;
    bool alive;
};

struct Bullet {
    int32_t x, y, vx, vy;
    int32_t angle;
    int32_t radius;
    int32_t life;
    bool fromSaucer;
    bool alive;
};

struct Saucer {
    int32_t x, y, vx, vy;
    int32_t radius;
    bool small;
    int32_t fireCooldown;
    int32_t driftTimer;
    bool alive;
};

struct Ship {
    int32_t x, y, vx, vy;
    int32_t angle;
    int32_t radius;
    bool canControl;
    int32_t fireCooldown;
    int32_t respawnTimer;
    int32_t invulnerableTimer;
    bool alive;
};

struct Simulation {
    Rng rng;
    Ship ship;
    Asteroid asteroids[ASTEROID_CAP];
    int32_t astCount;
    Bullet bullets[SHIP_BULLET_LIMIT];
    int32_t bulletCount;
    Saucer saucers[3];
    int32_t saucerCount;
    Bullet saucerBullets[SAUCER_BULLET_LIMIT];
    int32_t saucerBulletCount;
    int32_t score;
    int32_t lives;
    int32_t wave;
    int32_t nextExtraLifeScore;
    int32_t frameCount;
    int32_t timeSinceLastKill;
    int32_t saucerSpawnTimer;
    int32_t nextId;
    bool shipFireLatch;
    bool gameOver;
};

__device__ __forceinline__ int32_t simClamp(int32_t v, int32_t mn, int32_t mx) {
    return (v < mn) ? mn : (v > mx) ? mx : v;
}

__device__ __forceinline__ int32_t saucerWavePressurePct(const Simulation& sim) {
    return simClamp((sim.wave - 1) * 8, 0, 100);
}

__device__ __forceinline__ int32_t saucerLurkPressurePct(const Simulation& sim) {
    int32_t over = sim.timeSinceLastKill - LURK_TIME_THRESHOLD_FRAMES;
    if (over < 0) {
        over = 0;
    }
    int32_t v = (int32_t)((int64_t)over * 100 / (LURK_TIME_THRESHOLD_FRAMES * 2));
    return simClamp(v, 0, 100);
}

__device__ __forceinline__ int32_t saucerPressurePct(const Simulation& sim) {
    int32_t wp = saucerWavePressurePct(sim);
    int32_t lp = saucerLurkPressurePct(sim);
    int32_t v = wp + (lp * 50 / 100);
    return (v < 100) ? v : 100;
}

__device__ __forceinline__ void saucerFireCooldownRange(
    const Simulation& sim, bool small,
    int32_t& outMin, int32_t& outMax) {
    int32_t pressure = saucerPressurePct(sim);
    int32_t baseMin, baseMax, floorMin, floorMax;
    if (small) {
        baseMin = 42;
        baseMax = 68;
        floorMin = 22;
        floorMax = 40;
    } else {
        baseMin = 66;
        baseMax = 96;
        floorMin = 36;
        floorMax = 56;
    }

    int32_t mn = baseMin - ((baseMin - floorMin) * pressure / 100);
    int32_t mx = baseMax - ((baseMax - floorMax) * pressure / 100);
    if (mx <= mn) {
        mx = mn;
    }
    outMin = mn;
    outMax = mx;
}

__device__ __forceinline__ int32_t getSmallSaucerAimErrorBAM(const Simulation& sim) {
    int32_t pressure = saucerPressurePct(sim);
    int32_t baseError = 22;
    int32_t minError = 3;
    int32_t errorRange = baseError - minError;
    int32_t v = baseError - (int32_t)((int64_t)errorRange * pressure / 100);
    return simClamp(v, minError, baseError);
}

__device__ inline void simCreateAsteroid(
    Simulation& sim, int32_t size, int32_t x, int32_t y, Asteroid& out) {
    int32_t mn, mx;
    astSpeedRange(size, mn, mx);
    int32_t moveAngle = sim.rng.nextRange(0, 256);
    int32_t speed = sim.rng.nextRange(mn, mx);
    int32_t addMult = (sim.wave - 1) * 15;
    if (addMult > 128) {
        addMult = 128;
    }
    speed = speed + ((speed * addMult) >> 8);

    int32_t vx, vy;
    simVelocity(moveAngle, speed, vx, vy);

    int32_t startAngle = sim.rng.nextRange(0, 256);
    int32_t spin = sim.rng.nextRange(-3, 4);

    out.x = x;
    out.y = y;
    out.vx = vx;
    out.vy = vy;
    out.angle = startAngle;
    out.spin = spin;
    out.radius = astRadius(size);
    out.size = size;
    out.alive = true;
}

__device__ __forceinline__ void simQueueShipRespawn(Simulation& sim, int32_t delayFrames) {
    sim.ship.canControl = false;
    sim.ship.respawnTimer = delayFrames;
    sim.ship.vx = 0;
    sim.ship.vy = 0;
    sim.ship.fireCooldown = 0;
    sim.ship.invulnerableTimer = 0;
    sim.shipFireLatch = false;
}

__device__ inline int64_t simSpawnSafetyScore(
    const Simulation& sim, int32_t spawnX, int32_t spawnY, int64_t bestKnown) {
    int64_t minClearance = INT64_MAX;
    for (int i = 0; i < ASTEROID_CAP; i++) {
        if (!sim.asteroids[i].alive) {
            continue;
        }
        int64_t c = clearanceSqQ12_4(
            sim.asteroids[i].x, sim.asteroids[i].y, sim.asteroids[i].radius,
            spawnX, spawnY, sim.ship.radius);
        if (c < minClearance) {
            minClearance = c;
        }
        if (minClearance < bestKnown) {
            return minClearance;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (!sim.saucers[i].alive) {
            continue;
        }
        int64_t c = clearanceSqQ12_4(
            sim.saucers[i].x, sim.saucers[i].y, sim.saucers[i].radius,
            spawnX, spawnY, sim.ship.radius);
        if (c < minClearance) {
            minClearance = c;
        }
        if (minClearance < bestKnown) {
            return minClearance;
        }
    }

    for (int i = 0; i < SHIP_BULLET_LIMIT; i++) {
        if (!sim.bullets[i].alive) {
            continue;
        }
        int64_t c = clearanceSqQ12_4(
            sim.bullets[i].x, sim.bullets[i].y, sim.bullets[i].radius,
            spawnX, spawnY, sim.ship.radius);
        if (c < minClearance) {
            minClearance = c;
        }
        if (minClearance < bestKnown) {
            return minClearance;
        }
    }

    for (int i = 0; i < SAUCER_BULLET_LIMIT; i++) {
        if (!sim.saucerBullets[i].alive) {
            continue;
        }
        int64_t c = clearanceSqQ12_4(
            sim.saucerBullets[i].x, sim.saucerBullets[i].y, sim.saucerBullets[i].radius,
            spawnX, spawnY, sim.ship.radius);
        if (c < minClearance) {
            minClearance = c;
        }
        if (minClearance < bestKnown) {
            return minClearance;
        }
    }

    return minClearance;
}

__device__ inline void simFindBestShipSpawnPoint(
    const Simulation& sim, int32_t& bestX, int32_t& bestY) {
    int32_t centerX = WORLD_WIDTH_Q12_4 / 2;
    int32_t centerY = WORLD_HEIGHT_Q12_4 / 2;
    int32_t minX = SHIP_RESPAWN_EDGE_PADDING_Q12_4;
    int32_t maxX = WORLD_WIDTH_Q12_4 - SHIP_RESPAWN_EDGE_PADDING_Q12_4;
    int32_t minY = SHIP_RESPAWN_EDGE_PADDING_Q12_4;
    int32_t maxY = WORLD_HEIGHT_Q12_4 - SHIP_RESPAWN_EDGE_PADDING_Q12_4;
    int64_t bestSafetyScore = INT64_MIN;
    int64_t bestCenterDistance = INT64_MAX;
    bestX = centerX;
    bestY = centerY;
    for (int32_t y = minY; y <= maxY; y += SHIP_RESPAWN_GRID_STEP_Q12_4) {
        for (int32_t x = minX; x <= maxX; x += SHIP_RESPAWN_GRID_STEP_Q12_4) {
            int64_t safety = simSpawnSafetyScore(sim, x, y, bestSafetyScore);
            int64_t cdist = collisionDistSqQ12_4(x, y, centerX, centerY);
            if (safety > bestSafetyScore ||
                (safety == bestSafetyScore && cdist < bestCenterDistance)) {
                bestX = x;
                bestY = y;
                bestSafetyScore = safety;
                bestCenterDistance = cdist;
            }
        }
    }
}

__device__ inline void simSpawnShipAtBestOpenPoint(Simulation& sim) {
    int32_t spawnX, spawnY;
    simFindBestShipSpawnPoint(sim, spawnX, spawnY);
    sim.ship.x = spawnX;
    sim.ship.y = spawnY;
    sim.ship.vx = 0;
    sim.ship.vy = 0;
    sim.ship.angle = SHIP_FACING_UP_BAM;
    sim.ship.canControl = true;
    sim.ship.invulnerableTimer = SHIP_SPAWN_INVULNERABLE_FRAMES;
}

__device__ __forceinline__ void simAddScore(Simulation& sim, int32_t points) {
    sim.score += points;
    if (sim.score >= sim.nextExtraLifeScore) {
        sim.lives++;
        sim.nextExtraLifeScore += EXTRA_LIFE_SCORE_STEP;
    }
}

__device__ inline void simSpawnWave(Simulation& sim) {
    sim.wave++;
    sim.timeSinceLastKill = 0;

    int32_t largeCount = waveLargeAsteroidCount(sim.wave);
    int32_t avoidX = WORLD_WIDTH_Q12_4 / 2;
    int32_t avoidY = WORLD_HEIGHT_Q12_4 / 2;

    for (int i = 0; i < ASTEROID_CAP; i++) {
        sim.asteroids[i].alive = false;
    }
    sim.astCount = 0;

    for (int i = 0; i < largeCount; i++) {
        int32_t x = sim.rng.nextRange(0, WORLD_WIDTH_Q12_4);
        int32_t y = sim.rng.nextRange(0, WORLD_HEIGHT_Q12_4);
        int guard = 0;
        while (guard < 20 &&
               collisionDistSqQ12_4(x, y, avoidX, avoidY) < WAVE_SAFE_DIST_SQ_Q24_8) {
            x = sim.rng.nextRange(0, WORLD_WIDTH_Q12_4);
            y = sim.rng.nextRange(0, WORLD_HEIGHT_Q12_4);
            guard++;
        }
        simCreateAsteroid(sim, AST_LARGE, x, y, sim.asteroids[sim.astCount]);
        sim.astCount++;
    }

    simQueueShipRespawn(sim, 0);
    simSpawnShipAtBestOpenPoint(sim);
}

__device__ inline void simSpawnSaucer(Simulation& sim) {
    if (sim.saucerCount >= 3) {
        return;
    }

    bool enterFromLeft = (sim.rng.next() & 1) == 0;
    bool isLurking = sim.timeSinceLastKill > LURK_TIME_THRESHOLD_FRAMES;
    int32_t smallPct = isLurking ? 90 : (sim.score > 4000 ? 70 : 22);
    bool small = (int32_t)(sim.rng.next() % 100) < smallPct;
    int32_t speedQ8_8 = small ? SAUCER_SPEED_SMALL_Q8_8 : SAUCER_SPEED_LARGE_Q8_8;
    int32_t startX = enterFromLeft ? SAUCER_START_X_LEFT_Q12_4 : SAUCER_START_X_RIGHT_Q12_4;
    int32_t startY = sim.rng.nextRange(SAUCER_START_Y_MIN_Q12_4, SAUCER_START_Y_MAX_Q12_4);
    int32_t vy = sim.rng.nextRange(-94, 95);
    int32_t cooldownMin, cooldownMax;
    saucerFireCooldownRange(sim, small, cooldownMin, cooldownMax);
    int32_t fireCooldown = sim.rng.nextRange(cooldownMin, cooldownMax + 1);
    int32_t driftTimer = sim.rng.nextRange(48, 120);
    int slot = -1;
    for (int i = 0; i < 3; i++) {
        if (!sim.saucers[i].alive) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return;
    }

    sim.saucers[slot].x = startX;
    sim.saucers[slot].y = startY;
    sim.saucers[slot].vx = enterFromLeft ? speedQ8_8 : -speedQ8_8;
    sim.saucers[slot].vy = vy;
    sim.saucers[slot].radius = small ? SAUCER_RADIUS_SMALL : SAUCER_RADIUS_LARGE;
    sim.saucers[slot].small = small;
    sim.saucers[slot].fireCooldown = fireCooldown;
    sim.saucers[slot].driftTimer = driftTimer;
    sim.saucers[slot].alive = true;
    sim.saucerCount++;
}

__device__ inline void simSpawnSaucerBullet(Simulation& sim, int32_t saucerIdx) {
    if (sim.saucerBulletCount >= SAUCER_BULLET_LIMIT) {
        return;
    }

    const Saucer& saucer = sim.saucers[saucerIdx];
    int32_t shotAngle;

    if (saucer.small) {
        int32_t dx = shortDX(saucer.x, sim.ship.x);
        int32_t dy = shortDY(saucer.y, sim.ship.y);
        int32_t targetAngle = simAtan2(dy, dx);
        int32_t errorBAM = getSmallSaucerAimErrorBAM(sim);
        shotAngle = (targetAngle + sim.rng.nextRange(-errorBAM, errorBAM + 1)) & 0xff;
    } else {
        shotAngle = sim.rng.nextRange(0, 256);
    }

    int32_t vx, vy;
    simVelocity(shotAngle, SAUCER_BULLET_SPEED_Q8_8, vx, vy);
    int32_t offDx, offDy;
    simDisplace(shotAngle, saucer.radius + 4, offDx, offDy);
    int32_t startX = simWrapX(saucer.x + offDx);
    int32_t startY = simWrapY(saucer.y + offDy);

    int slot = -1;
    for (int i = 0; i < SAUCER_BULLET_LIMIT; i++) {
        if (!sim.saucerBullets[i].alive) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return;
    }

    sim.saucerBullets[slot].x = startX;
    sim.saucerBullets[slot].y = startY;
    sim.saucerBullets[slot].vx = vx;
    sim.saucerBullets[slot].vy = vy;
    sim.saucerBullets[slot].angle = shotAngle;
    sim.saucerBullets[slot].radius = 2;
    sim.saucerBullets[slot].life = SAUCER_BULLET_LIFETIME_FRAMES;
    sim.saucerBullets[slot].fromSaucer = true;
    sim.saucerBullets[slot].alive = true;
    sim.saucerBulletCount++;
}

__device__ inline void simSpawnShipBullet(Simulation& sim) {
    if (sim.bulletCount >= SHIP_BULLET_LIMIT) {
        return;
    }

    int32_t dx, dy;
    simDisplace(sim.ship.angle, sim.ship.radius + 6, dx, dy);
    int32_t startX = simWrapX(sim.ship.x + dx);
    int32_t startY = simWrapY(sim.ship.y + dy);

    int32_t absVx = sim.ship.vx < 0 ? -sim.ship.vx : sim.ship.vx;
    int32_t absVy = sim.ship.vy < 0 ? -sim.ship.vy : sim.ship.vy;
    int32_t shipSpeedApprox = ((absVx + absVy) * 3) >> 2;
    int32_t bulletSpeedQ8_8 = SHIP_BULLET_SPEED_Q8_8 + ((shipSpeedApprox * 89) >> 8);

    int32_t bvx, bvy;
    simVelocity(sim.ship.angle, bulletSpeedQ8_8, bvx, bvy);

    int slot = -1;
    for (int i = 0; i < SHIP_BULLET_LIMIT; i++) {
        if (!sim.bullets[i].alive) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return;
    }

    sim.bullets[slot].x = startX;
    sim.bullets[slot].y = startY;
    sim.bullets[slot].vx = sim.ship.vx + bvx;
    sim.bullets[slot].vy = sim.ship.vy + bvy;
    sim.bullets[slot].angle = sim.ship.angle;
    sim.bullets[slot].radius = 2;
    sim.bullets[slot].life = SHIP_BULLET_LIFETIME_FRAMES;
    sim.bullets[slot].fromSaucer = false;
    sim.bullets[slot].alive = true;
    sim.bulletCount++;
}

__device__ inline void simDestroyShip(Simulation& sim) {
    simQueueShipRespawn(sim, SHIP_RESPAWN_FRAMES);
    sim.lives--;
    if (sim.lives <= 0) {
        sim.ship.canControl = false;
        sim.ship.respawnTimer = 99999;
    }
}

__device__ inline int32_t simDestroyAsteroid(
    Simulation& sim, int32_t idx, bool awardScore, int32_t aliveCount) {
    if (!sim.asteroids[idx].alive) {
        return aliveCount;
    }

    sim.asteroids[idx].alive = false;
    aliveCount = (aliveCount > 0) ? aliveCount - 1 : 0;

    if (awardScore) {
        sim.timeSinceLastKill = 0;
        if (sim.asteroids[idx].size == AST_LARGE) {
            simAddScore(sim, SCORE_LARGE_ASTEROID);
        } else if (sim.asteroids[idx].size == AST_MEDIUM) {
            simAddScore(sim, SCORE_MEDIUM_ASTEROID);
        } else {
            simAddScore(sim, SCORE_SMALL_ASTEROID);
        }
    }

    if (sim.asteroids[idx].size == AST_SMALL) {
        return aliveCount;
    }

    int32_t childSize = (sim.asteroids[idx].size == AST_LARGE) ? AST_MEDIUM : AST_SMALL;
    int32_t freeSlots = (ASTEROID_CAP - aliveCount > 0) ? ASTEROID_CAP - aliveCount : 0;
    int32_t splitCount = (freeSlots < 2) ? freeSlots : 2;
    int32_t parentVx = sim.asteroids[idx].vx;
    int32_t parentVy = sim.asteroids[idx].vy;
    int32_t parentX = sim.asteroids[idx].x;
    int32_t parentY = sim.asteroids[idx].y;

    int freeSlotArr[2] = {-1, -1};
    int foundSlots = 0;
    for (int i = ASTEROID_CAP - 1; i >= 0 && foundSlots < splitCount; i--) {
        if (!sim.asteroids[i].alive) {
            freeSlotArr[foundSlots++] = i;
        }
    }

    for (int ci = 0; ci < foundSlots; ci++) {
        int slot = (foundSlots == 2) ? freeSlotArr[1 - ci] : freeSlotArr[ci];
        simCreateAsteroid(sim, childSize, parentX, parentY, sim.asteroids[slot]);
        sim.asteroids[slot].vx += (parentVx * 46) >> 8;
        sim.asteroids[slot].vy += (parentVy * 46) >> 8;
        aliveCount++;
    }

    return aliveCount;
}

__device__ inline void simUpdateCounts(Simulation& sim) {
    {
        int w = 0;
        for (int i = 0; i < ASTEROID_CAP; i++) {
            if (sim.asteroids[i].alive) {
                if (i != w) {
                    sim.asteroids[w] = sim.asteroids[i];
                }
                w++;
            }
        }
        for (int i = w; i < ASTEROID_CAP; i++) {
            sim.asteroids[i].alive = false;
        }
        sim.astCount = w;
    }
    {
        int w = 0;
        for (int i = 0; i < SHIP_BULLET_LIMIT; i++) {
            if (sim.bullets[i].alive) {
                if (i != w) {
                    sim.bullets[w] = sim.bullets[i];
                }
                w++;
            }
        }
        for (int i = w; i < SHIP_BULLET_LIMIT; i++) {
            sim.bullets[i].alive = false;
        }
        sim.bulletCount = w;
    }
    {
        int w = 0;
        for (int i = 0; i < 3; i++) {
            if (sim.saucers[i].alive) {
                if (i != w) {
                    sim.saucers[w] = sim.saucers[i];
                }
                w++;
            }
        }
        for (int i = w; i < 3; i++) {
            sim.saucers[i].alive = false;
        }
        sim.saucerCount = w;
    }
    {
        int w = 0;
        for (int i = 0; i < SAUCER_BULLET_LIMIT; i++) {
            if (sim.saucerBullets[i].alive) {
                if (i != w) {
                    sim.saucerBullets[w] = sim.saucerBullets[i];
                }
                w++;
            }
        }
        for (int i = w; i < SAUCER_BULLET_LIMIT; i++) {
            sim.saucerBullets[i].alive = false;
        }
        sim.saucerBulletCount = w;
    }
}

__device__ inline void simUpdateShip(Simulation& sim, uint8_t inp) {
    Ship& ship = sim.ship;
    bool fire = (inp & INPUT_FIRE) != 0;
    bool left = (inp & INPUT_LEFT) != 0;
    bool right = (inp & INPUT_RIGHT) != 0;
    bool thrust = (inp & INPUT_THRUST) != 0;

    if (ship.fireCooldown > 0) {
        ship.fireCooldown--;
    }
    if (!fire) {
        sim.shipFireLatch = false;
    }

    if (!ship.canControl) {
        if (ship.respawnTimer > 0) {
            ship.respawnTimer--;
        }
        if (ship.respawnTimer <= 0) {
            simSpawnShipAtBestOpenPoint(sim);
        }
        if (fire) {
            sim.shipFireLatch = true;
        }
        return;
    }

    if (ship.invulnerableTimer > 0) {
        ship.invulnerableTimer--;
    }
    if (left) {
        ship.angle = (ship.angle - SHIP_TURN_SPEED_BAM) & 0xff;
    }
    if (right) {
        ship.angle = (ship.angle + SHIP_TURN_SPEED_BAM) & 0xff;
    }
    if (thrust) {
        int32_t accelVx = (simCos(ship.angle) * SHIP_THRUST_Q8_8) >> 14;
        int32_t accelVy = (simSin(ship.angle) * SHIP_THRUST_Q8_8) >> 14;
        ship.vx += accelVx;
        ship.vy += accelVy;
    }

    ship.vx = simApplyDrag(ship.vx);
    ship.vy = simApplyDrag(ship.vy);
    simClampSpeed(ship.vx, ship.vy, SHIP_MAX_SPEED_SQ_Q16_16);

    bool firePressedThisFrame = fire && !sim.shipFireLatch;
    if (firePressedThisFrame && ship.fireCooldown <= 0 && sim.bulletCount < SHIP_BULLET_LIMIT) {
        simSpawnShipBullet(sim);
        ship.fireCooldown = SHIP_BULLET_COOLDOWN_FRAMES;
    }
    if (fire) {
        sim.shipFireLatch = true;
    }

    ship.x = simWrapX(ship.x + (ship.vx >> 4));
    ship.y = simWrapY(ship.y + (ship.vy >> 4));
}

__device__ inline void simUpdateAsteroids(Simulation& sim) {
    for (int i = 0; i < ASTEROID_CAP; i++) {
        if (!sim.asteroids[i].alive) {
            continue;
        }
        sim.asteroids[i].x = simWrapX(sim.asteroids[i].x + (sim.asteroids[i].vx >> 4));
        sim.asteroids[i].y = simWrapY(sim.asteroids[i].y + (sim.asteroids[i].vy >> 4));
        sim.asteroids[i].angle = (sim.asteroids[i].angle + sim.asteroids[i].spin) & 0xff;
    }
}

__device__ inline void simUpdateBullets(Simulation& sim) {
    for (int i = 0; i < SHIP_BULLET_LIMIT; i++) {
        if (!sim.bullets[i].alive) {
            continue;
        }
        sim.bullets[i].life--;
        if (sim.bullets[i].life <= 0) {
            sim.bullets[i].alive = false;
            continue;
        }
        sim.bullets[i].x = simWrapX(sim.bullets[i].x + (sim.bullets[i].vx >> 4));
        sim.bullets[i].y = simWrapY(sim.bullets[i].y + (sim.bullets[i].vy >> 4));
    }
}

__device__ inline void simUpdateSaucerBullets(Simulation& sim) {
    for (int i = 0; i < SAUCER_BULLET_LIMIT; i++) {
        if (!sim.saucerBullets[i].alive) {
            continue;
        }
        sim.saucerBullets[i].life--;
        if (sim.saucerBullets[i].life <= 0) {
            sim.saucerBullets[i].alive = false;
            continue;
        }
        sim.saucerBullets[i].x = simWrapX(
            sim.saucerBullets[i].x + (sim.saucerBullets[i].vx >> 4));
        sim.saucerBullets[i].y = simWrapY(
            sim.saucerBullets[i].y + (sim.saucerBullets[i].vy >> 4));
    }
}

__device__ inline void simUpdateSaucers(Simulation& sim) {
    if (sim.saucerSpawnTimer > 0) {
        sim.saucerSpawnTimer--;
    }

    bool isLurking = sim.timeSinceLastKill > LURK_TIME_THRESHOLD_FRAMES;
    int32_t spawnThreshold = isLurking ? LURK_SAUCER_SPAWN_FAST_FRAMES : 0;
    int32_t maxSaucers = maxSaucersForWave(sim.wave);

    if (sim.saucerCount < maxSaucers && sim.saucerSpawnTimer <= spawnThreshold) {
        simSpawnSaucer(sim);
        int32_t waveMultPct = 100 - (sim.wave - 1) * 8;
        if (waveMultPct < 40) {
            waveMultPct = 40;
        }
        int32_t spawnMin = (int32_t)((int64_t)SAUCER_SPAWN_MIN_FRAMES * waveMultPct / 100);
        int32_t spawnMax = (int32_t)((int64_t)SAUCER_SPAWN_MAX_FRAMES * waveMultPct / 100);
        if (isLurking) {
            sim.saucerSpawnTimer = sim.rng.nextRange(
                LURK_SAUCER_SPAWN_FAST_FRAMES,
                LURK_SAUCER_SPAWN_FAST_FRAMES + 120);
        } else {
            sim.saucerSpawnTimer = sim.rng.nextRange(spawnMin, spawnMax);
        }
    }

    for (int i = 0; i < 3; i++) {
        Saucer& saucer = sim.saucers[i];
        if (!saucer.alive) {
            continue;
        }

        saucer.x = saucer.x + (saucer.vx >> 4);
        saucer.y = simWrapY(saucer.y + (saucer.vy >> 4));

        if (saucer.x < SAUCER_CULL_MIN_X_Q12_4 || saucer.x > SAUCER_CULL_MAX_X_Q12_4) {
            saucer.alive = false;
            sim.saucerCount--;
            continue;
        }

        if (saucer.driftTimer > 0) {
            saucer.driftTimer--;
        }
        if (saucer.driftTimer <= 0) {
            saucer.driftTimer = sim.rng.nextRange(48, 120);
            saucer.vy = sim.rng.nextRange(-163, 164);
        }

        if (saucer.fireCooldown > 0) {
            saucer.fireCooldown--;
        }
        if (saucer.fireCooldown <= 0) {
            simSpawnSaucerBullet(sim, i);
            int32_t cdMin, cdMax;
            saucerFireCooldownRange(sim, saucer.small, cdMin, cdMax);
            saucer.fireCooldown = sim.rng.nextRange(cdMin, cdMax + 1);
        }
    }
}

__device__ inline void simHandleCollisions(Simulation& sim) {
    int32_t aliveAsteroids = sim.astCount;

    for (int bi = 0; bi < SHIP_BULLET_LIMIT; bi++) {
        if (aliveAsteroids == 0) {
            break;
        }
        if (!sim.bullets[bi].alive) {
            continue;
        }
        for (int ai = 0; ai < ASTEROID_CAP; ai++) {
            if (!sim.asteroids[ai].alive) {
                continue;
            }
            if (collidesQ12_4(
                    sim.bullets[bi].x, sim.bullets[bi].y, sim.bullets[bi].radius,
                    sim.asteroids[ai].x, sim.asteroids[ai].y, sim.asteroids[ai].radius)) {
                sim.bullets[bi].alive = false;
                sim.bulletCount = (sim.bulletCount > 0) ? sim.bulletCount - 1 : 0;
                aliveAsteroids = simDestroyAsteroid(sim, ai, true, aliveAsteroids);
                break;
            }
        }
    }

    for (int bi = 0; bi < SAUCER_BULLET_LIMIT; bi++) {
        if (aliveAsteroids == 0) {
            break;
        }
        if (!sim.saucerBullets[bi].alive) {
            continue;
        }
        for (int ai = 0; ai < ASTEROID_CAP; ai++) {
            if (!sim.asteroids[ai].alive) {
                continue;
            }
            if (collidesQ12_4(
                    sim.saucerBullets[bi].x, sim.saucerBullets[bi].y, sim.saucerBullets[bi].radius,
                    sim.asteroids[ai].x, sim.asteroids[ai].y, sim.asteroids[ai].radius)) {
                sim.saucerBullets[bi].alive = false;
                sim.saucerBulletCount = (sim.saucerBulletCount > 0) ? sim.saucerBulletCount - 1 : 0;
                aliveAsteroids = simDestroyAsteroid(sim, ai, false, aliveAsteroids);
                break;
            }
        }
    }

    for (int bi = 0; bi < SHIP_BULLET_LIMIT; bi++) {
        if (!sim.bullets[bi].alive) {
            continue;
        }
        for (int si = 0; si < 3; si++) {
            if (!sim.saucers[si].alive) {
                continue;
            }
            if (collidesQ12_4(
                    sim.bullets[bi].x, sim.bullets[bi].y, sim.bullets[bi].radius,
                    sim.saucers[si].x, sim.saucers[si].y, sim.saucers[si].radius)) {
                sim.bullets[bi].alive = false;
                sim.bulletCount = (sim.bulletCount > 0) ? sim.bulletCount - 1 : 0;
                sim.saucers[si].alive = false;
                sim.saucerCount = (sim.saucerCount > 0) ? sim.saucerCount - 1 : 0;
                simAddScore(sim, sim.saucers[si].small ? SCORE_SMALL_SAUCER : SCORE_LARGE_SAUCER);
                break;
            }
        }
    }

    if (aliveAsteroids > 0) {
        for (int si = 0; si < 3; si++) {
            if (!sim.saucers[si].alive) {
                continue;
            }
            for (int ai = 0; ai < ASTEROID_CAP; ai++) {
                if (!sim.asteroids[ai].alive) {
                    continue;
                }
                if (collidesQ12_4(
                        sim.saucers[si].x, sim.saucers[si].y, sim.saucers[si].radius,
                        sim.asteroids[ai].x, sim.asteroids[ai].y, sim.asteroids[ai].radius)) {
                    sim.saucers[si].alive = false;
                    sim.saucerCount = (sim.saucerCount > 0) ? sim.saucerCount - 1 : 0;
                    break;
                }
            }
        }
    }

    if (!sim.ship.canControl || sim.ship.invulnerableTimer > 0) {
        return;
    }

    if (aliveAsteroids > 0) {
        for (int ai = 0; ai < ASTEROID_CAP; ai++) {
            if (!sim.asteroids[ai].alive) {
                continue;
            }
            int32_t adjustedRadius = (sim.asteroids[ai].radius * 225) >> 8;
            if (collidesQ12_4(
                    sim.ship.x, sim.ship.y, sim.ship.radius,
                    sim.asteroids[ai].x, sim.asteroids[ai].y, adjustedRadius)) {
                simDestroyShip(sim);
                return;
            }
        }
    }

    for (int bi = 0; bi < SAUCER_BULLET_LIMIT; bi++) {
        if (!sim.saucerBullets[bi].alive) {
            continue;
        }
        if (collidesQ12_4(
                sim.ship.x, sim.ship.y, sim.ship.radius,
                sim.saucerBullets[bi].x, sim.saucerBullets[bi].y, sim.saucerBullets[bi].radius)) {
            sim.saucerBullets[bi].alive = false;
            sim.saucerBulletCount = (sim.saucerBulletCount > 0) ? sim.saucerBulletCount - 1 : 0;
            simDestroyShip(sim);
            return;
        }
    }

    for (int si = 0; si < 3; si++) {
        if (!sim.saucers[si].alive) {
            continue;
        }
        if (collidesQ12_4(
                sim.ship.x, sim.ship.y, sim.ship.radius,
                sim.saucers[si].x, sim.saucers[si].y, sim.saucers[si].radius)) {
            sim.saucers[si].alive = false;
            sim.saucerCount = (sim.saucerCount > 0) ? sim.saucerCount - 1 : 0;
            simDestroyShip(sim);
            return;
        }
    }
}

__device__ inline void simStep(Simulation& sim, uint8_t inp) {
    sim.frameCount++;
    simUpdateShip(sim, inp);
    simUpdateAsteroids(sim);
    simUpdateBullets(sim);
    simUpdateSaucers(sim);
    simUpdateSaucerBullets(sim);
    simHandleCollisions(sim);
    simUpdateCounts(sim);
    sim.timeSinceLastKill++;
    if (sim.astCount == 0 && sim.saucerCount == 0) {
        simSpawnWave(sim);
    }
}

__device__ inline void simInit(Simulation& sim, uint32_t seed) {
    memset(&sim, 0, sizeof(Simulation));
    sim.rng.seed(seed);
    sim.score = 0;
    sim.lives = STARTING_LIVES;
    sim.wave = 0;
    sim.nextExtraLifeScore = EXTRA_LIFE_SCORE_STEP;
    sim.frameCount = 0;
    sim.timeSinceLastKill = 0;
    sim.gameOver = false;
    sim.shipFireLatch = false;
    sim.nextId = 1;

    for (int i = 0; i < ASTEROID_CAP; i++) {
        sim.asteroids[i].alive = false;
    }
    for (int i = 0; i < SHIP_BULLET_LIMIT; i++) {
        sim.bullets[i].alive = false;
    }
    for (int i = 0; i < SAUCER_BULLET_LIMIT; i++) {
        sim.saucerBullets[i].alive = false;
    }
    for (int i = 0; i < 3; i++) {
        sim.saucers[i].alive = false;
    }

    sim.astCount = 0;
    sim.bulletCount = 0;
    sim.saucerCount = 0;
    sim.saucerBulletCount = 0;
    sim.ship.x = WORLD_WIDTH_Q12_4 / 2;
    sim.ship.y = WORLD_HEIGHT_Q12_4 / 2;
    sim.ship.vx = 0;
    sim.ship.vy = 0;
    sim.ship.angle = SHIP_FACING_UP_BAM;
    sim.ship.radius = SHIP_RADIUS;
    sim.ship.canControl = true;
    sim.ship.fireCooldown = 0;
    sim.ship.respawnTimer = 0;
    sim.ship.invulnerableTimer = SHIP_SPAWN_INVULNERABLE_FRAMES;
    sim.ship.alive = true;

    simSpawnWave(sim);
    {
        int32_t waveMultPct = 100 - (sim.wave - 1) * 8;
        if (waveMultPct < 40) {
            waveMultPct = 40;
        }
        int32_t spawnMin = (int32_t)((int64_t)SAUCER_SPAWN_MIN_FRAMES * waveMultPct / 100);
        int32_t spawnMax = (int32_t)((int64_t)SAUCER_SPAWN_MAX_FRAMES * waveMultPct / 100);
        sim.saucerSpawnTimer = sim.rng.nextRange(spawnMin, spawnMax);
    }
}
