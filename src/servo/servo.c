#include "servo.h"

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void servo_init(servo_t *s, float min_deg, float max_deg,
                uint16_t min_us, uint16_t max_us, float max_speed_dps) {
    s->min_deg = min_deg;
    s->max_deg = max_deg;
    s->min_us = min_us;
    s->max_us = max_us;
    s->max_speed_dps = max_speed_dps;
    s->current_deg = min_deg;
}

uint16_t servo_angle_to_us(const servo_t *s, float deg) {
    deg = clampf(deg, s->min_deg, s->max_deg);
    float frac = (deg - s->min_deg) / (s->max_deg - s->min_deg);
    return (uint16_t)(s->min_us + frac * (s->max_us - s->min_us) + 0.5f);
}

float servo_step(servo_t *s, float target_deg, float dt_s) {
    target_deg = clampf(target_deg, s->min_deg, s->max_deg);
    float delta = target_deg - s->current_deg;
    float max_move = s->max_speed_dps * dt_s;
    if (delta > max_move)
        s->current_deg += max_move;
    else if (delta < -max_move)
        s->current_deg -= max_move;
    else
        s->current_deg = target_deg;  /* close enough to land exactly, no overshoot */
    return s->current_deg;
}
