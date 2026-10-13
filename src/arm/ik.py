import math

# placeholder link lengths in cm; real values come from the arm we end up with
L1, L2 = 10.0, 10.0


def forward(q1, q2, l1=L1, l2=L2):
    """Joint angles (radians) -> where the tip ends up."""
    x = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    y = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
    return x, y


def inverse(x, y, l1=L1, l2=L2, elbow_up=True):
    """Target point -> joint angles (radians), or None if the arm can't reach it."""
    d2 = x * x + y * y
    cos_q2 = (d2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(cos_q2) > 1:
        return None  # too far or too close
    q2 = math.acos(cos_q2)
    if not elbow_up:
        q2 = -q2
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1, q2


if __name__ == "__main__":
    # round trip: pick angles, get the tip, solve back, expect the same tip
    for q1_deg, q2_deg in [(30, 60), (90, 45), (10, 120), (45, -50)]:
        q1, q2 = math.radians(q1_deg), math.radians(q2_deg)
        x, y = forward(q1, q2)
        sol = inverse(x, y, elbow_up=q2 >= 0)
        fx, fy = forward(*sol)
        assert abs(fx - x) < 1e-9 and abs(fy - y) < 1e-9
        print(f"({x:6.2f}, {y:6.2f}) -> {math.degrees(sol[0]):6.1f}, {math.degrees(sol[1]):6.1f} deg")

    assert inverse(25, 0) is None       # farther than L1 + L2 = 20
    assert inverse(0, 0) is not None    # fully folded back is reachable
    print("all ik tests passed")
