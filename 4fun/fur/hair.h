#pragma once

#include "glm_helper.cuh"
#include "simulation_parameters.h"

struct Hair {
    vec3 pos[HAIR_SEGMENTS];
    vec3 old_pos[HAIR_SEGMENTS];
    vec3 vel[HAIR_SEGMENTS];
    vec3 color;
    vec3 anchor_direction;
};
