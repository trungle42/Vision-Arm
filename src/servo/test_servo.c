#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "servo.h"

static int near(float a, float b) { return fabsf(a - b) < 1e-3f; }

int main(void) {
    servo_t s;
    servo_init(&s, 0.0f, 180.0f, 500, 2500, 90.0f);

    /* angle -> pulse width */
    assert(servo_angle_to_us(&s, 0.0f) == 500);
    assert(servo_angle_to_us(&s, 90.0f) == 1500);
    assert(servo_angle_to_us(&s, 180.0f) == 2500);
    assert(servo_angle_to_us(&s, -10.0f) == 500);    /* clamped low */
    assert(servo_angle_to_us(&s, 200.0f) == 2500);   /* clamped high */

    /* speed limit: 90 deg/s, 0.5 s steps = 45 deg per step */
    assert(near(servo_step(&s, 90.0f, 0.5f), 45.0f));
    assert(near(servo_step(&s, 90.0f, 0.5f), 90.0f));
    assert(near(servo_step(&s, 90.0f, 0.5f), 90.0f));  /* already there, no overshoot */

    /* moving back down, and a target past the range gets clamped */
    assert(near(servo_step(&s, 0.0f, 0.5f), 45.0f));
    assert(near(servo_step(&s, 500.0f, 10.0f), 180.0f));

    /* a small remaining distance lands exactly on the target */
    servo_init(&s, 0.0f, 180.0f, 500, 2500, 90.0f);
    assert(near(servo_step(&s, 10.0f, 1.0f), 10.0f));

    printf("all servo tests passed\n");
    return 0;
}
