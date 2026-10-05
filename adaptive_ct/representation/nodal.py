"""Continuous (shared-corner) trilinear field on an octree.

Every leaf corner sits on the integer lattice of the finest level. Each lattice position stores one
value (a node). A corner lying inside a face or an edge of a larger leaf (not on that leaf's own
corners) is not free: its value is the larger leaf's trilinear interpolation at that point, so the
field stays continuous across leaves of different sizes. Constraints referring to other constrained
corners are substituted until only free nodes remain.

Result: leaf corner coefficients  c = P n  with a sparse P (rows: leaf*8 + corner, columns: free
nodes), nonnegative weights and unit row sums, so n >= 0 gives a nonnegative field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_OFFSETS = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.long)


@dataclass
class NodalMap:
    P: torch.Tensor          # sparse CSR (leaves*8, free_nodes)
    Pt: torch.Tensor         # sparse CSR transpose
    rep_slot: torch.Tensor   # (free_nodes,) one leaf-corner slot holding each free node's value
    positions: torch.Tensor  # (free_nodes, 3) lattice coordinates

    @property
    def free_nodes(self) -> int:
        return int(self.rep_slot.numel())

    def to_corners(self, n):
        return torch.sparse.mm(self.P, n[:, None])[:, 0]

    def to_nodes(self, g):
        return torch.sparse.mm(self.Pt, g[:, None])[:, 0]


def _trilinear(u):
    a = torch.stack([1 - u[:, 0], u[:, 0]], -1)
    b = torch.stack([1 - u[:, 1], u[:, 1]], -1)
    c = torch.stack([1 - u[:, 2], u[:, 2]], -1)
    return (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]).reshape(-1, 8)


@torch.no_grad()
def build_nodal_map(model) -> NodalMap:
    if not bool(torch.all(model.leaf_degrees == 1)):
        raise ValueError("the shared-corner field needs every leaf to be trilinear")
    dev = model.leaf_levels.device
    offsets = _OFFSETS.to(dev)
    top = len(model.level_shapes) - 1
    span = [int(v) + 1 for v in model.level_shapes[top]]

    def key(pos):
        return (pos[..., 0] * span[1] + pos[..., 1]) * span[2] + pos[..., 2]

    scale = (2 ** (top - model.leaf_levels)).long()
    corners = (model.leaf_coords[:, None, :] + offsets[None]) * scale[:, None, None]  # (N, 8, 3)
    keys, inverse = torch.unique(key(corners).reshape(-1), return_inverse=True)
    pos = corners.reshape(-1, 3)[torch.zeros(keys.numel(), dtype=torch.long, device=dev).scatter_(0, inverse, torch.arange(inverse.numel(), device=dev))]
    npos = int(keys.numel())

    constrained = torch.zeros(npos, dtype=torch.bool, device=dev)
    rows, cols, vals = [], [], []
    for level in range(top):
        s = 2 ** (top - level)
        shape = torch.tensor(model.level_shapes[level], device=dev)
        linear, leaf_ids = model._level_leaf_lookup(level)
        if linear.numel() == 0:
            continue
        todo = torch.nonzero(~constrained & torch.any(pos % s != 0, dim=1)).flatten()
        if todo.numel() == 0:
            continue
        p = pos[todo]
        on_grid = (p % s == 0).long()
        found = torch.full((todo.numel(),), -1, dtype=torch.long, device=dev)
        cell_found = torch.zeros(todo.numel(), 3, dtype=torch.long, device=dev)
        for d in offsets:
            cell = torch.div(p, s, rounding_mode="floor") - d[None] * on_grid
            inside = torch.all((cell >= 0) & (cell < shape[None]), dim=1)
            lin = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
            at = torch.searchsorted(linear, lin).clamp_max(linear.numel() - 1)
            hit = inside & (linear[at] == lin) & (found < 0)
            found[hit] = leaf_ids[at[hit]]
            cell_found[hit] = cell[hit]
        ok = found >= 0
        if not bool(ok.any()):
            continue
        idx, p, cell = todo[ok], p[ok], cell_found[ok]
        w = _trilinear((p - cell * s).double() / s)
        ref = torch.searchsorted(keys, key((cell[:, None, :] + offsets[None]) * s))
        keep = w > 1e-12
        rows.append(idx[:, None].expand(-1, 8)[keep])
        cols.append(ref[keep])
        vals.append(w[keep])
        constrained[idx] = True

    free = torch.nonzero(~constrained).flatten()
    rows.append(free)
    cols.append(free)
    vals.append(torch.ones(free.numel(), dtype=torch.float64, device=dev))
    S = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals), (npos, npos)).coalesce()
    for _ in range(2 * top + 2):  # substitute constrained references until every column is a free node
        col_constrained = constrained[S.indices()[1]]
        if not bool(col_constrained.any()):
            break
        S = torch.sparse.mm(S, S).coalesce()
    else:
        raise RuntimeError("constraint substitution did not terminate")
    node_id = torch.full((npos,), -1, dtype=torch.long, device=dev)
    node_id[free] = torch.arange(free.numel(), device=dev)
    r, c = S.indices()
    pos_rows = torch.sparse_coo_tensor(torch.stack([r, node_id[c]]), S.values().float(), (npos, free.numel())).coalesce().to_sparse_csr()
    # leaf-corner slot j uses the row of its lattice position
    crow, colv, valv = pos_rows.crow_indices(), pos_rows.col_indices(), pos_rows.values()
    start, count = crow[inverse], crow[inverse + 1] - crow[inverse]
    slot = torch.repeat_interleave(torch.arange(inverse.numel(), device=dev), count)
    within = torch.arange(int(count.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(count, 0) - count, count)
    gather = torch.repeat_interleave(start, count) + within
    P = torch.sparse_coo_tensor(torch.stack([slot, colv[gather]]), valv[gather], (inverse.numel(), free.numel())).coalesce()
    first = torch.full((npos,), inverse.numel(), dtype=torch.long, device=dev).scatter_reduce_(0, inverse, torch.arange(inverse.numel(), device=dev), "amin")
    return NodalMap(P=P.to_sparse_csr(), Pt=P.t().coalesce().to_sparse_csr(), rep_slot=first[free], positions=pos[free])
