"""Sparse octree of typed leaves (constant / trilinear / step), laid out for the CUDA traversal.

Leaf order: all trilinear leaves, then all step leaves, then all constant leaves; inside each type in Z order
(Morton code of the cell's corner at the finest level, so leaves close in space are close in memory).
The type and the value offset of leaf i follow from its index and the three type counts:
  trilinear  i < n_tri               values[8 i : 8 i + 8]            (one 32-byte aligned block)
  step       n_tri <= i < n_tri + n_step   values[8 n_tri + 2 j : + 2], j = i - n_tri;  geometry row j
  constant   otherwise               values[8 n_tri + 2 n_step + k],  k = i - n_tri - n_step
Values are the physical attenuation (>= 0).  A step leaf is (1 - H) c0 + H c1,
H = sigmoid((n . (u - 1/2) - d) / w), u = leaf-local coordinates in [0, 1]^3; w = 0 is a hard step.
Its geometry row holds n (3), d, w and three padding floats (one 32-byte record).
Tree: root[cell of the root grid] and child_table[internal node, child slot] are int32; a value >= 0 is a leaf
index, a value < 0 is the internal node -1 - value.  Child slot = 4 x + 2 y + z of the child's parity bits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from adaptive_ct.core.projector import bernstein_octree_integrate, has_bernstein_native

TYPE_NAMES = ("constant", "trilinear", "step")  # type id = index
VALUE_COUNT = (1, 8, 2)  # stored values per leaf, by type id
GEOMETRY_WIDTH = 8
_ORDER_RANK = (2, 0, 1)  # position of each type id in the leaf order: trilinear, step, constant
_MISSING = -(2**31)


def step_integral(geometry: torch.Tensor, s_start: torch.Tensor, s_stop: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Exact integral of H over straight segments, given s = n . (u - 1/2) at both ends (same formula as the CUDA kernel)."""
    s0, s1, w = s_start - geometry[:, 3], s_stop - geometry[:, 3], geometry[:, 4]
    hard = torch.where((s0 > 0) & (s1 > 0), length, torch.where((s0 <= 0) & (s1 <= 0), torch.zeros_like(length),
                                                                 length * torch.maximum(s0, s1) / (s1 - s0).abs().clamp_min(1e-30)))
    ws = w.clamp_min(1e-30)
    a0, a1 = s0 / ws, s1 / ws
    close = (a1 - a0).abs() < 1e-3
    smooth = torch.where(close, length * torch.sigmoid(0.5 * (a0 + a1)),
                         length * (F.softplus(a1) - F.softplus(a0)) / torch.where(close, torch.ones_like(a0), a1 - a0))
    return torch.where(w > 0, smooth, hard)


def step_H(geometry: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """H of step geometry rows (n, >= 5) at leaf-local points u (n, 3)."""
    s = ((u - 0.5) * geometry[:, :3]).sum(1) - geometry[:, 3]
    w = geometry[:, 4]
    return torch.where(w > 0, torch.sigmoid(s / w.clamp_min(1e-30)), (s > 0).to(s.dtype))


def trilinear_weights(u: torch.Tensor) -> torch.Tensor:
    """(n, 8) corner weights at leaf-local points u (n, 3); corner index 4 i + 2 j + k."""
    p = torch.stack([1 - u, u], -1)
    return (p[:, 0, :, None, None] * p[:, 1, None, :, None] * p[:, 2, None, None, :]).reshape(-1, 8)


def pad_geometry(g: torch.Tensor) -> torch.Tensor:
    """(n, 5) n, d, w -> (n, GEOMETRY_WIDTH)."""
    return F.pad(g[:, :5], (0, GEOMETRY_WIDTH - 5))


def morton(coords: torch.Tensor, bits: int) -> torch.Tensor:
    """Z-order code of nonnegative integer coordinates (n, 3)."""
    code = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    for b in range(bits):
        for axis in range(3):
            code |= ((coords[:, axis] >> b) & 1) << (3 * b + 2 - axis)
    return code


@dataclass(frozen=True)
class OctreeStats:
    leaf_cells: int
    active_by_level: tuple[int, ...]
    max_depth: int
    value_count: int
    leaf_types: tuple[str, ...]
    leaf_type_counts: tuple[int, ...]


def _shape3(value: int | Sequence[int]) -> tuple[int, int, int]:
    shape = (int(value),) * 3 if isinstance(value, int) else tuple(int(v) for v in value)
    if len(shape) != 3:
        raise ValueError(f"Expected an integer or a 3-value shape, got {value!r}.")
    return shape


def _grid_coords(shape, device=None) -> torch.Tensor:
    axes = [torch.arange(v, dtype=torch.long, device=device) for v in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


class BernsteinOctree(nn.Module):
    representation = "bernstein_octree"

    def __init__(self, *, levels: Iterable[int | Sequence[int]], leaf_types: Sequence[str] = TYPE_NAMES,
                 cuda_integrator: bool = True, max_leaf_count: int | None = None):
        super().__init__()
        self.level_shapes = [_shape3(v) for v in levels]
        if not self.level_shapes or any(c <= 0 for s in self.level_shapes for c in s):
            raise ValueError("Octree resolutions must be positive.")
        for parent, child in zip(self.level_shapes, self.level_shapes[1:]):
            if any(child[a] != 2 * parent[a] for a in range(3)):
                raise ValueError(f"Octree levels must refine dyadically; got parent={parent}, child={child}.")
        self.leaf_types = tuple(str(t) for t in leaf_types)
        if not self.leaf_types or any(t not in TYPE_NAMES for t in self.leaf_types):
            raise ValueError(f"leaf_types must be a non-empty subset of {list(TYPE_NAMES)}.")
        self.l0_shape = self.level_shapes[0]
        self.cuda_integrator = bool(cuda_integrator)
        self.max_leaf_count = int(max_leaf_count) if max_leaf_count is not None else None
        self.last_split_report: dict = {}  # what the last split_leaves_batch changed beyond the function (2 % rule)
        self._morton_bits = max(1, math.ceil(math.log2(max(self.level_shapes[-1]))))
        coords = _grid_coords(self.l0_shape)
        n = coords.shape[0]
        if self.max_leaf_count is not None and self.max_leaf_count < n:
            raise ValueError("max_leaf_count cannot be smaller than the number of root cells.")
        self.values = nn.Parameter(torch.zeros(0), requires_grad=False)
        self.register_buffer("cells", torch.zeros((0, 4), dtype=torch.int32))  # level, x, y, z
        self.register_buffer("type_counts", torch.zeros(3, dtype=torch.long))  # n_tri, n_step, n_const
        self.register_buffer("step_geometry", torch.zeros((0, GEOMETRY_WIDTH)))
        self.register_buffer("root", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.register_buffer("child_table", torch.zeros((0, 8), dtype=torch.int32), persistent=False)
        self._assign(torch.zeros(n, dtype=torch.long), coords, torch.zeros(n, dtype=torch.long), torch.zeros(n),
                     torch.zeros((n, GEOMETRY_WIDTH)))

    # ------------------------------------------------------------------ layout
    @property
    def leaf_count(self) -> int:
        return int(self.cells.shape[0])

    @property
    def leaf_levels(self) -> torch.Tensor:
        return self.cells[:, 0].long()

    @property
    def leaf_coords(self) -> torch.Tensor:
        return self.cells[:, 1:].long()

    def leaf_type(self) -> torch.Tensor:
        """(N,) type ids (index into TYPE_NAMES)."""
        i = torch.arange(self.leaf_count, device=self.cells.device)
        return torch.where(i < self.n_tri, 1, torch.where(i < self.n_tri + self.n_step, 2, 0))

    def value_offsets(self) -> torch.Tensor:
        """(N,) offset of each leaf's first value."""
        i = torch.arange(self.leaf_count, device=self.cells.device)
        base = 8 * self.n_tri
        return torch.where(i < self.n_tri, 8 * i, torch.where(i < self.n_tri + self.n_step, base + 2 * (i - self.n_tri),
                                                                base + 2 * self.n_step + (i - self.n_tri - self.n_step)))

    def leaf_mean(self) -> torch.Tensor:
        """(N,) mean of each leaf's stored values (summaries and the viewer)."""
        counts = torch.tensor(VALUE_COUNT, device=self.cells.device)[self.leaf_type()]
        owner = torch.repeat_interleave(torch.arange(self.leaf_count, device=self.cells.device), counts)
        # values are laid out leaf by leaf in index order, so the owner of value k is the k-th entry of `owner`
        return torch.zeros(self.leaf_count, device=self.cells.device).index_add(0, owner, self.values.detach()) / counts

    def geometry(self, ids: torch.Tensor) -> torch.Tensor:
        """(n, GEOMETRY_WIDTH) geometry rows of leaves `ids` (zeros for leaves that are not steps)."""
        j = ids - self.n_tri
        step = (j >= 0) & (j < self.n_step)
        out = torch.zeros((ids.numel(), GEOMETRY_WIDTH), device=self.cells.device)
        out[step] = self.step_geometry[j[step]]
        return out

    def leaf_values(self, ids: torch.Tensor) -> torch.Tensor:
        """Values of leaves `ids`, concatenated in that order."""
        counts = torch.tensor(VALUE_COUNT, device=ids.device)[self.leaf_type()[ids]]
        starts = torch.cumsum(counts, 0) - counts
        idx = torch.repeat_interleave(self.value_offsets()[ids] - starts, counts) + torch.arange(int(counts.sum()), device=ids.device)
        return self.values.detach()[idx]

    @torch.no_grad()
    def _assign(self, levels, coords, types, values, geometry) -> None:
        """Set all leaves (any order; `values` concatenated in that order, `geometry` (N, GEOMETRY_WIDTH) per leaf,
        rows of non-step leaves ignored), sort them into the canonical layout and rebuild the tree."""
        device = self.cells.device
        levels, coords, types = levels.to(device).long(), coords.to(device).long(), types.to(device).long()
        values, geometry = values.to(device).float(), geometry.to(device).float()
        counts = torch.tensor(VALUE_COUNT, device=device)[types]
        corner = coords << (len(self.level_shapes) - 1 - levels)[:, None]
        rank = torch.tensor(_ORDER_RANK, device=device)[types]
        order = torch.argsort((rank << (3 * self._morton_bits)) | morton(corner, self._morton_bits))
        new_counts = counts[order]
        starts = torch.cumsum(counts, 0) - counts
        new_starts = torch.cumsum(new_counts, 0) - new_counts
        idx = torch.repeat_interleave(starts[order] - new_starts, new_counts) + torch.arange(int(new_counts.sum()), device=device)
        types = types[order]
        self.values = nn.Parameter(values[idx].clamp_min(0).contiguous(), requires_grad=False)
        self.cells = torch.cat([levels[order, None], coords[order]], 1).to(torch.int32).contiguous()
        self.type_counts = torch.stack([(types == 1).sum(), (types == 2).sum(), (types == 0).sum()]).long()
        self.step_geometry = geometry[order][types == 2].contiguous()
        self._rebuild_tree()

    @torch.no_grad()
    def _rebuild_tree(self) -> None:
        self.n_tri, self.n_step, self.n_const = (int(v) for v in self.type_counts.tolist())
        device = self.cells.device
        levels, coords = self.leaf_levels, self.leaf_coords
        L = len(self.level_shapes)
        linear = lambda c, s: (c[:, 0] * s[1] + c[:, 1]) * s[2] + c[:, 2]
        keys, base = [], [0]
        for l in range(L):  # sorted linear keys of the internal nodes of every level
            deeper = levels > l
            k = torch.unique(linear(coords[deeper] >> (levels[deeper] - l)[:, None], self.level_shapes[l])) if l + 1 < L else \
                torch.zeros(0, dtype=torch.long, device=device)
            keys.append(k)
            base.append(base[-1] + int(k.numel()))
        root = torch.full((math.prod(self.l0_shape),), _MISSING, dtype=torch.long, device=device)
        children = torch.full((base[-1], 8), _MISSING, dtype=torch.long, device=device)
        ids = torch.arange(self.leaf_count, device=device)
        for l in range(L):
            leaf = levels == l
            entries = [(coords[leaf], ids[leaf]), (None, -1 - (base[l] + torch.arange(keys[l].numel(), device=device)))]
            for c, value in entries:
                if c is None:  # internal nodes of this level, from their keys
                    s = self.level_shapes[l]
                    k = keys[l]
                    c = torch.stack([k // (s[1] * s[2]), (k // s[2]) % s[1], k % s[2]], 1)
                if l == 0:
                    root[linear(c, self.l0_shape)] = value
                elif c.numel():
                    parent = base[l - 1] + torch.searchsorted(keys[l - 1], linear(c >> 1, self.level_shapes[l - 1]))
                    children[parent, (c[:, 0] & 1) * 4 + (c[:, 1] & 1) * 2 + (c[:, 2] & 1)] = value
        if bool((root == _MISSING).any()) or bool((children == _MISSING).any()):
            raise ValueError("Invalid octree: a cell is neither a leaf nor an internal node.")
        self.root = root.to(torch.int32).contiguous()
        self.child_table = children.to(torch.int32).contiguous()

    def resolve_leaf_ids(self, points: torch.Tensor) -> torch.Tensor:
        """Leaf index containing each point (-1 outside [-1, 1]^3)."""
        p = points.reshape(-1, 3)
        inside = torch.all((p >= -1.0) & (p <= 1.0), dim=1)
        v = None
        for l, shape in enumerate(self.level_shapes):
            s = torch.tensor(shape, device=p.device)
            c = torch.minimum(torch.floor((p + 1.0) * 0.5 * s).long().clamp_min(0), s - 1)
            if l == 0:
                v = self.root[(c[:, 0] * shape[1] + c[:, 1]) * shape[2] + c[:, 2]].long()
                continue
            inner = v < 0
            if not bool(inner.any()):
                break
            slot = (c[inner, 0] & 1) * 4 + (c[inner, 1] & 1) * 2 + (c[inner, 2] & 1)
            v[inner] = self.child_table[-1 - v[inner], slot].long()
        return torch.where(inside, v, -1).reshape(points.shape[:-1])

    def ids_of_cells(self, cells: torch.Tensor) -> torch.Tensor:
        """Leaf indices of cells (n, 4) = level, x, y, z (each must be a leaf)."""
        shapes = torch.tensor(self.level_shapes, dtype=torch.float32, device=cells.device)[cells[:, 0].long()]
        return self.resolve_leaf_ids((cells[:, 1:].float() + 0.5) / shapes * 2 - 1)

    def uses_cuda(self) -> bool:
        return self.cuda_integrator and self.values.is_cuda and has_bernstein_native()

    def find_leaf(self, key: Sequence[int]) -> int | None:
        """Index of the leaf at cell (level, x, y, z), or None."""
        level, *xyz = (int(v) for v in key)
        if not 0 <= level < len(self.level_shapes) or any(not 0 <= c < n for c, n in zip(xyz, self.level_shapes[level])):
            return None
        centre = (torch.tensor(xyz, dtype=torch.float32, device=self.cells.device) + 0.5) / torch.tensor(self.level_shapes[level], device=self.cells.device) * 2 - 1
        leaf = int(self.resolve_leaf_ids(centre[None])[0])
        return leaf if leaf >= 0 and self.cells[leaf].tolist() == [level, *xyz] else None

    # ------------------------------------------------------------------ evaluation
    def local_coords(self, ids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        shapes = torch.tensor(self.level_shapes, dtype=points.dtype, device=points.device)[self.leaf_levels[ids]]
        return (points + 1.0) * 0.5 * shapes - self.leaf_coords[ids].to(points.dtype)

    def forward_mu_with_leaf_ids(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = points.reshape(-1, 3)
        leaf = self.resolve_leaf_ids(flat)
        out = torch.zeros(flat.shape[0], device=flat.device)
        sel = torch.nonzero(leaf >= 0).flatten()
        ids = leaf[sel]
        u = self.local_coords(ids, flat[sel]).clamp(0.0, 1.0)
        off = self.value_offsets()[ids]
        kind = self.leaf_type()[ids]
        v = self.values
        out = out.index_put((sel[kind == 0],), v[off[kind == 0]])
        t = kind == 1
        out = out.index_put((sel[t],), (trilinear_weights(u[t]) * v[off[t, None] + torch.arange(8, device=flat.device)]).sum(1))
        t = kind == 2
        H = step_H(self.step_geometry[ids[t] - self.n_tri], u[t])
        out = out.index_put((sel[t],), (1 - H) * v[off[t]] + H * v[off[t] + 1])
        return out.reshape(points.shape[:-1]), leaf.reshape(points.shape[:-1])

    def forward_mu(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_mu_with_leaf_ids(points)[0]

    def decoded_at_resolution(self, resolution: int | tuple[int, int, int], *, chunk: int = 1 << 20) -> torch.Tensor:
        shape = _shape3(resolution)
        device = self.cells.device
        scale = torch.tensor(shape, dtype=torch.float32, device=device)
        total, ny, nz = math.prod(shape), shape[1], shape[2]
        out = []
        for start in range(0, total, int(chunk)):
            i = torch.arange(start, min(start + int(chunk), total), device=device)
            c = torch.stack([i // (ny * nz), (i // nz) % ny, i % nz], 1).float()
            out.append(self.forward_mu(-1.0 + (c + 0.5) * 2.0 / scale))
        return torch.cat(out).reshape(shape)

    # ------------------------------------------------------------------ rays (PyTorch reference of the CUDA traversal)
    @staticmethod
    def _ray_frame(ray_batch, device, dtype):
        a = -ray_batch.angles.to(device=device, dtype=dtype)
        u = -1.0 + (ray_batch.cols.to(device=device, dtype=dtype) + 0.5) * 2.0 / float(ray_batch.detector_w)
        z = -1.0 + (ray_batch.rows.to(device=device, dtype=dtype) + 0.5) * 2.0 / float(ray_batch.detector_h)
        direction = torch.stack([torch.cos(a), torch.sin(a)], 1)
        base = torch.stack([-torch.sin(a), torch.cos(a)], 1) * u[:, None]
        return direction, base, z

    def _parallel_ray_segments(self, ray_batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ray, leaf, [t0, t1]) segments, cut at every finest-level cell boundary."""
        device, dtype, eps = self.cells.device, torch.float32, 1e-8
        direction, base, z = self._ray_frame(ray_batch, device, dtype)
        t_min = torch.full((direction.shape[0],), -1e20, device=device)
        t_max = torch.full_like(t_min, 1e20)
        valid = torch.ones_like(t_min, dtype=torch.bool)
        cuts = []
        for axis in range(2):
            d, b = direction[:, axis], base[:, axis]
            parallel = d.abs() < eps
            valid &= ~parallel | ((b >= -1) & (b <= 1))
            den = torch.where(parallel, torch.ones_like(d), d)
            t0, t1 = (-1 - b) / den, (1 - b) / den
            t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
            t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
            n = self.level_shapes[-1][axis]
            bounds = -1.0 + torch.arange(1, n, device=device, dtype=dtype) * (2.0 / n)
            cuts.append(((bounds[None] - b[:, None]) / den[:, None], parallel))
        valid &= t_max > t_min
        cand = [t_min[:, None], t_max[:, None]]
        for c, parallel in cuts:
            keep = ~parallel[:, None] & (c > t_min[:, None] + eps) & (c < t_max[:, None] - eps)
            cand.append(torch.where(keep, c, torch.full_like(c, float("inf"))))
        t = torch.sort(torch.cat(cand, 1), 1).values
        starts, stops = t[:, :-1], t[:, 1:]
        ok = valid[:, None] & torch.isfinite(starts) & torch.isfinite(stops) & (stops > starts + eps)
        ray_ids = torch.arange(ray_batch.num_rays, device=device)[:, None].expand_as(starts)[ok]
        starts, stops = starts[ok], stops[ok]
        mid = 0.5 * (starts + stops)
        xy = base[ray_ids] + direction[ray_ids] * mid[:, None]
        leaf = self.resolve_leaf_ids(torch.stack([xy[:, 0], xy[:, 1], z[ray_ids]], 1))
        keep = leaf >= 0
        return ray_ids[keep], leaf[keep], torch.stack([starts[keep], stops[keep]], 1)

    def segment_local_coords(self, ray_batch, ray_ids: torch.Tensor, leaf_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Leaf-local coordinates of the points at ray parameter t."""
        direction, base, z = self._ray_frame(ray_batch, t.device, t.dtype)
        xy = base[ray_ids] + direction[ray_ids] * t[:, None]
        return self.local_coords(leaf_ids, torch.stack([xy[:, 0], xy[:, 1], z[ray_ids]], 1))

    def integrate_ray_batch(self, ray_batch) -> torch.Tensor:
        """Exact line integrals (CUDA traversal, or the PyTorch reference on CPU; differentiable in values and step geometry)."""
        if self.uses_cuda():
            return bernstein_octree_integrate(self.values, self.step_geometry, **self.tree_arguments(), rays=ray_batch)
        ray_ids, leaf, iv = self._parallel_ray_segments(ray_batch)
        u0 = self.segment_local_coords(ray_batch, ray_ids, leaf, iv[:, 0])
        u1 = self.segment_local_coords(ray_batch, ray_ids, leaf, iv[:, 1])
        length = iv[:, 1] - iv[:, 0]
        kind = self.leaf_type()[leaf]
        off = self.value_offsets()[leaf]
        v = self.values
        seg = torch.zeros_like(length)
        seg = seg.index_put((kind == 0,), v[off[kind == 0]] * length[kind == 0])
        t = kind == 1
        tri = sum(0.5 * trilinear_weights(u0[t] + (u1[t] - u0[t]) * q) for q in (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)))
        seg = seg.index_put((t,), length[t] * (tri * v[off[t, None] + torch.arange(8, device=v.device)]).sum(1))
        t = kind == 2
        g = self.step_geometry[leaf[t] - self.n_tri]
        I = step_integral(g, ((u0[t] - 0.5) * g[:, :3]).sum(1), ((u1[t] - 0.5) * g[:, :3]).sum(1), length[t])
        seg = seg.index_put((t,), v[off[t]] * (length[t] - I) + v[off[t] + 1] * I)
        return torch.zeros(ray_batch.num_rays, device=v.device).index_add(0, ray_ids, seg)

    def tree_arguments(self) -> dict:
        return {"root": self.root, "children": self.child_table, "n_tri": self.n_tri, "n_step": self.n_step,
                "level_shapes": self.level_shapes}

    # ------------------------------------------------------------------ structure changes
    def _rest(self, ids: torch.Tensor):
        """Leaves other than `ids`: (levels, coords, types, values, geometry)."""
        keep = torch.ones(self.leaf_count, dtype=torch.bool, device=self.cells.device)
        keep[ids] = False
        k = torch.nonzero(keep).flatten()
        return self.leaf_levels[k], self.leaf_coords[k], self.leaf_type()[k], self.leaf_values(k), self.geometry(k)

    def _replace(self, ids, levels, coords, types, values, geometry) -> None:
        rest = self._rest(ids)
        self._assign(*(torch.cat([a, b.to(a.device, a.dtype)]) for a, b in zip(rest, (levels, coords, types, values, geometry))))

    @torch.no_grad()
    def split_leaves_batch(self, leaf_ids) -> int:
        """Split leaves into 8 children of the same type, function preserving:
        constant children copy the parent; trilinear children take the parent's values at their corners;
        step children keep both values with geometry n' = n, d' = 2d - n . (o - 1/2), w' = 2w (o = child offset),
        except children where one side of the step fills < 2% of the volume: those become constant (mean value).
        Returns the number of split leaves (capped by max_leaf_count)."""
        device = self.cells.device
        ids = torch.unique(torch.as_tensor(leaf_ids, dtype=torch.long, device=device))
        ids = ids[(ids >= 0) & (ids < self.leaf_count)]
        if self.max_leaf_count is not None:
            ids = ids[: max(0, (self.max_leaf_count - self.leaf_count) // 7)]
        if ids.numel() == 0:
            return 0
        if bool((self.leaf_levels[ids] + 1 >= len(self.level_shapes)).any()):
            raise ValueError("At least one selected leaf is already at maximum octree depth.")
        kind = self.leaf_type()[ids]
        offsets = _grid_coords((2, 2, 2), device)  # child o, slot 4 x + 2 y + z
        off = self.value_offsets()[ids]
        v = self.values.detach()
        types = kind.repeat_interleave(8)
        geometry = torch.zeros((ids.numel(), 8, GEOMETRY_WIDTH), device=device)
        vals = torch.zeros((ids.numel(), 8, 8), device=device)  # up to 8 values per child
        c = kind == 0
        vals[c, :, 0] = v[off[c]][:, None]
        c = kind == 1
        parents = v[off[c, None] + torch.arange(8, device=device)].reshape(-1, 2, 2, 2)
        half = parents.new_tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # corner values at 0, 1/2, 1 per axis
        lattice = torch.einsum("ai,bj,ck,nijk->nabc", half, half, half, parents)
        vals[c] = torch.stack([lattice[:, x : x + 2, y : y + 2, z : z + 2].reshape(-1, 8) for x, y, z in offsets.tolist()], 1)
        c = kind == 2
        pair = torch.stack([v[off[c]], v[off[c] + 1]], 1)
        g = self.step_geometry[ids[c] - self.n_tri][:, None, :]
        shift = (g[..., :3] * (offsets[None].float() - 0.5)).sum(-1)
        cg = torch.cat([g[..., :3].expand(-1, 8, -1), (2 * g[..., 3] - shift)[..., None], (2 * g[..., 4]).expand(-1, 8)[..., None]], -1)
        # A child where one side of the step fills < 2% of its volume becomes a constant leaf (its mean value):
        # the value on the small side would hardly affect any ray and drift unconstrained in the solve.
        q = (torch.arange(4, device=device, dtype=torch.float32) + 0.5) / 4
        u = torch.stack(torch.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
        sv = ((u[None, None] - 0.5) * cg[..., None, :3]).sum(-1) - cg[..., 3:4]
        Hs = torch.where(cg[..., 4:5] > 0, torch.sigmoid(sv / cg[..., 4:5].clamp_min(1e-30)), (sv > 0).float())
        mean_h = Hs.mean(-1)
        flat = torch.minimum(mean_h, 1 - mean_h) < 0.02
        # what this lossy rule changes: |c1 - c0| times the deviation of H from its mean, over the flattened children
        jump = (pair[:, None, 1] - pair[:, None, 0]).abs()
        deviation = (Hs - mean_h[..., None]).abs()
        self.last_split_report = {
            "step_children": int(flat.numel()), "children_to_constant": int(flat.sum()),
            "max_abs_change": float((deviation.amax(-1) * jump)[flat].max()) if bool(flat.any()) else 0.0,
            "rms_change": float(((deviation ** 2).mean(-1) * jump ** 2)[flat].mean().sqrt()) if bool(flat.any()) else 0.0}
        vals[c, :, 0] = torch.where(flat, (1 - mean_h) * pair[:, None, 0] + mean_h * pair[:, None, 1], pair[:, None, 0].expand(-1, 8))
        vals[c, :, 1] = pair[:, None, 1]
        geometry[c] = pad_geometry(cg.reshape(-1, 5)).reshape(-1, 8, GEOMETRY_WIDTH)
        kc = types.reshape(-1, 8)
        kc[c] = torch.where(flat, 0, 2)
        types = kc.reshape(-1)
        counts = torch.tensor(VALUE_COUNT, device=device)[types]
        values = vals.reshape(-1, 8)[torch.arange(8, device=device)[None] < counts[:, None]]
        self._replace(ids, (self.leaf_levels[ids] + 1).repeat_interleave(8), (self.leaf_coords[ids][:, None] * 2 + offsets[None]).reshape(-1, 3),
                      types, values, geometry.reshape(-1, GEOMETRY_WIDTH))
        return int(ids.numel())

    @torch.no_grad()
    def set_types(self, leaf_ids, types, values: torch.Tensor, geometry: torch.Tensor) -> None:
        """Replace leaves (same cells) by leaves of type `types` with the given values (concatenated in leaf order)
        and geometry rows (n, >= 5; used by step leaves)."""
        device = self.cells.device
        ids = torch.as_tensor(leaf_ids, dtype=torch.long, device=device)
        types = torch.as_tensor(types, dtype=torch.long, device=device)
        if ids.numel() == 0:
            return
        allowed = torch.tensor([t in self.leaf_types for t in TYPE_NAMES], device=device)
        if not bool(allowed[types].all()):
            raise ValueError(f"leaf types outside {self.leaf_types}")
        self._replace(ids, self.leaf_levels[ids], self.leaf_coords[ids], types, values, pad_geometry(geometry.to(device).float()))

    @torch.no_grad()
    def to_trilinear(self, leaf_ids) -> None:
        """Constant leaves -> trilinear leaves with all 8 corners equal to the constant (same function)."""
        ids = torch.unique(torch.as_tensor(leaf_ids, dtype=torch.long, device=self.cells.device))
        ids = ids[self.leaf_type()[ids] == 0]
        c = self.values.detach()[self.value_offsets()[ids]]
        self.set_types(ids, torch.ones_like(ids), c.repeat_interleave(8), torch.zeros((ids.numel(), 5)))

    def stats(self) -> OctreeStats:
        by_level = tuple(int((self.cells[:, 0] == l).sum()) for l in range(len(self.level_shapes)))
        return OctreeStats(
            leaf_cells=self.leaf_count,
            active_by_level=by_level,
            max_depth=max((l for l, n in enumerate(by_level) if n), default=0),
            value_count=int(self.values.numel()),
            leaf_types=self.leaf_types,
            leaf_type_counts=(self.n_const, self.n_tri, self.n_step),
        )

    # ------------------------------------------------------------------ checkpoints
    def prepare_sparse_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Resize the leaf arrays to a checkpoint before load_state_dict."""
        device = self.cells.device
        self.values = nn.Parameter(torch.empty_like(state_dict["values"], device=device), requires_grad=False)
        for name in ("cells", "type_counts", "step_geometry"):
            setattr(self, name, torch.empty_like(state_dict[name], device=device))

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._rebuild_tree()
        return result
