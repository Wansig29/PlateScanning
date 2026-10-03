"""Traffic scenes the plate tracker must keep apart: one ID per vehicle, no swaps, no splits."""
import random

import pytest

from platescanner.vision.tracker import PlateTracker

W, H, FPS = 80, 28, 15


def mv(x0, y0, vx, vy=0.0):
    return lambda t: (int(x0 + vx * t), int(y0 + vy * t), W, H)


def jitter(path, amount=5, seed=3):
    rng = random.Random(seed)
    return lambda t: (lambda b: None if b is None else
                      (b[0] + rng.randint(-amount, amount), b[1] + rng.randint(-amount, amount),
                       b[2] + rng.randint(-amount, amount), b[3]))(path(t))


def id_changes(paths, frames=40):
    """For each true vehicle: how many times its track id changed."""
    tracker = PlateTracker(max_age=0.8)
    seen = [[] for _ in paths]
    for f in range(frames):
        t = f / FPS
        boxes, owner = [], []
        for i, p in enumerate(paths):
            b = p(t)
            if b:
                boxes.append(b)
                owner.append(i)
        matched, _ = tracker.update(boxes, t)
        by_det = {d: tk.track_id for tk, d in matched}
        for d, i in enumerate(owner):
            seen[i].append(by_det[d])
    # Two vehicles must also never share an id.
    assert len({v[0] for v in seen}) == len(seen), "two vehicles got the same track"
    return [len(set(v)) - 1 for v in seen]


SCENES = {
    "bumper to bumper": lambda: [mv(100, 300, 300), mv(-100, 300, 300)],
    "plates almost touching": lambda: [mv(100, 300, 300), mv(-5, 300, 300)],
    "two lanes side by side": lambda: [mv(100, 300, 300), mv(100, 360, 300)],
    "opposite directions, adjacent lanes": lambda: [mv(100, 300, 300), mv(700, 340, -300)],
    "fast car": lambda: [mv(0, 300, 900)],
    "fast overtakes slow, adjacent lanes": lambda: [mv(0, 300, 900), mv(150, 345, 150)],
    "fast overtakes slow, same row": lambda: [mv(0, 300, 900), mv(150, 300, 150)],
    "diagonal crossing": lambda: [mv(100, 200, 250, 150), mv(100, 520, 250, -150)],
    "stopped car, another passes in front": lambda: [mv(200, 300, 0), mv(0, 320, 500)],
    "jittery boxes, bumper to bumper": lambda: [jitter(mv(100, 300, 300)), jitter(mv(-100, 300, 300), seed=4)],
    "jittery boxes, opposite lanes": lambda: [jitter(mv(100, 300, 300)), jitter(mv(700, 330, -300), seed=4)],
    "three-frame detector dropout": lambda: [lambda t: mv(100, 300, 300)(t) if not 0.7 < t < 0.9 else None],
}


@pytest.mark.parametrize("name", SCENES)
def test_each_vehicle_keeps_one_track(name):
    assert id_changes(SCENES[name]()) == [0] * len(SCENES[name]())
