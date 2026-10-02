"""Joint-action copy of a multi-agent dataset: `action` becomes [own action, partner action] per step.

train.py reads the `action` column (and only that column is packed by frameskip), so a joint-action
model trains on a copy whose `action` holds the egocentric joint action: agent i's own action first,
then the other agent's, as stored in collect.py's `joint_action` (agent-index order). The original
own action is kept as `action_self`. Every other column is copied unchanged (2 agents only).

    python scripts/make_joint_dataset.py multipusht_2a_coop.h5 multipusht_2a_coop_heldout.h5
    # -> datasets/multipusht_2a_coop_joint.h5, datasets/multipusht_2a_coop_heldout_joint.h5
"""

import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (Blosc filter for the pixel chunks)
import numpy as np

root = Path(os.environ.get("STABLEWM_HOME", Path(__file__).resolve().parents[1] / "data")) / "datasets"

for name in sys.argv[1:]:
    src, dst = root / name, root / name.replace(".h5", "_joint.h5")
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        joint, agent, own = f["joint_action"][:], f["agent_idx"][:], f["action"][:]
        assert joint.shape[1] == 4, "2 agents only"
        ego = np.where(agent[:, None] == 0, joint, joint[:, [2, 3, 0, 1]])
        assert np.allclose(ego[:, :2], own, equal_nan=True), "own half must equal the stored action"
        for key in f:
            if key != "action":
                f.copy(f[key], g, name=key)  # raw chunk copy, filters kept
        g.create_dataset("action", data=ego.astype(np.float32), chunks=(1000, 4))
        g.create_dataset("action_self", data=own, chunks=(1000, 2))
        for key in ("ep_len", "ep_offset", "episode_idx", "scene_idx"):  # a copy read too early (NFS) gets zeros here
            assert np.array_equal(g[key][:], f[key][:]), f"{key} differs from the source"
        assert g["ep_len"][:].sum() == len(ego) > 0, "episode index does not cover the rows"
    print(f"{dst}: action {ego.shape} = [self, partner]")
