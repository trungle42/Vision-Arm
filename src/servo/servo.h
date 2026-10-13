#ifndef SERVO_H
#define SERVO_H

#include <stdint.h>

/* Hobby servo: angle -> PWM pulse width, plus a speed limit so it does not jerk. */
typedef struct {
    float min_deg, max_deg;   /* mechanical range */
    uint16_t min_us, max_us;  /* pulse width at min_deg and max_deg */
    float max_speed_dps;      /* fastest allowed motion, degrees per second */
    float current_deg;        /* where we think the servo is */
} servo_t;

void servo_init(servo_t *s, float min_deg, float max_deg,
                uint16_t min_us, uint16_t max_us, float max_speed_dps);

/* Pulse width in microseconds for an angle. Angles outside the range are clamped. */
uint16_t servo_angle_to_us(const servo_t *s, float deg);

/* Move current_deg toward target_deg by at most max_speed_dps * dt_s. Returns the new angle. */
float servo_step(servo_t *s, float target_deg, float dt_s);

#endif
