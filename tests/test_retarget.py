import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts.retarget import kabsch, smooth_weights, pose_bones, donor_motion


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(30, 3))
    R = Rot.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    t = np.array([1.0, 2.0, -3.0])
    R2, t2 = kabsch(P, P @ R.T + t)
    assert np.allclose(R2, R, atol=1e-9) and np.allclose(t2, t, atol=1e-9)


def test_weights_rows_sum_to_one():
    hard = np.array([0, 0, 1, 1, 2])
    idx = np.array([[0, 1, 2], [2, 3, 4]])
    w = smooth_weights(hard, idx, 3, iterations=3)
    assert np.allclose(w.sum(1), 1) and (w >= 0).all()


def test_zero_motion_leaves_rest_pose():
    rest = np.random.default_rng(2).normal(size=(10, 3))
    w = np.zeros((10, 2)); w[:5, 0] = 1; w[5:, 1] = 1
    joints = {"torso": np.zeros(3), "arm": np.ones(3)}
    out = pose_bones(rest, w, ["torso", "arm"], {"torso": None, "arm": "torso"}, joints,
                     {"torso": np.zeros(3), "arm": np.zeros(3)}, np.zeros(3))
    assert np.allclose(out, rest)


def test_child_bone_follows_parent_rotation():
    rest = np.array([[1.0, 0, 0], [2.0, 0, 0]])
    w = np.array([[1, 0], [0, 1.0]])
    joints = {"torso": np.zeros(3), "arm": np.array([2.0, 0, 0])}
    rv = {"torso": np.array([0, 0, np.pi / 2]), "arm": np.zeros(3)}
    out = pose_bones(rest, w, ["torso", "arm"], {"torso": None, "arm": "torso"}, joints, rv, np.zeros(3))
    assert np.allclose(out, [[0, 1, 0], [0, 2, 0]], atol=1e-9)


def test_donor_motion_identity_at_rest():
    rest = np.random.default_rng(3).normal(size=(12, 3))
    lab = np.array([0] * 6 + [1] * 6)
    m = donor_motion(np.stack([rest, rest]), rest, lab)
    assert np.allclose(m["rotvec"], 0) and np.allclose(m["trans"], 0)


def test_soft_limit_is_identity_for_small_and_bounded_for_large():
    from scripts.golem_retarget import soft_limit
    small = np.array([0.02, 0, 0])
    assert np.allclose(soft_limit(small, 90), small, atol=1e-3)
    big = soft_limit(np.array([0, 0, 6.0]), 90)
    assert np.linalg.norm(big) <= np.radians(90) + 1e-9


def test_keypose_animation_hits_keys_and_loops():
    from scripts.golem_retarget import keypose_animation, golem_labels
    rng = np.random.default_rng(5)
    rest = rng.uniform([-1, 0, -0.4], [1, 2.4, 1.8], size=(60, 3))
    idx = rng.integers(0, 60, size=(40, 3))
    keys = [(0.0, {}, (0, 0, 0)), (0.5, {"torso": (30, 0, 0)}, (0, 0, 0))]
    a = keypose_animation(rest, idx, keys, 20, loop=True, ground=False)
    assert np.allclose(a[0], rest, atol=1e-9)                  # key 0 is the rest pose
    assert not np.allclose(a[10], rest)                        # key 1 moves the torso
    assert np.abs(a[0] - a[-1]).max() < np.abs(a[0] - a[10]).max()   # last frame is next to the first
