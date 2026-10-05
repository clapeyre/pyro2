"""Float64 CTU advection, preserving pyro2's indexing and operation order.

The numerical formulas follow pyro2 (BSD-3-Clause); see LICENSE at the repository root.
Periodic and outflow boundaries on Cartesian, constant-velocity scalar advection are supported.
"""
import math

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def mc(dl: wp.float64, dr: wp.float64, dc: wp.float64):
    d1 = wp.float64(2.0) * dr
    if wp.abs(dl) < wp.abs(dr):
        d1 = wp.float64(2.0) * dl
    value = d1
    if wp.abs(dc) < wp.abs(d1):
        value = dc
    if dl * dr <= wp.float64(0.0):
        value = wp.float64(0.0)
    return value


@wp.kernel
def fill_ghosts(a: wp.array2d(dtype=wp.float64), nx: int, ny: int, ng: int,
                periodic_x: int, periodic_y: int):
    i, j = wp.tid()
    if i < ng or i >= ng + nx or j < ng or j >= ng + ny:
        ii = wp.clamp(i, ng, ng + nx - 1)
        jj = wp.clamp(j, ng, ng + ny - 1)
        if periodic_x != 0:
            ii = (i - ng + nx * ng) % nx + ng
        if periodic_y != 0:
            jj = (j - ng + ny * ng) % ny + ng
        # Both indices refer to interior cells, including at corners. Ghost
        # threads never read other ghost threads' writes in this launch.
        a[i, j] = a[ii, jj]


@wp.kernel
def slopes(a: wp.array2d(dtype=wp.float64), sx: wp.array2d(dtype=wp.float64),
           sy: wp.array2d(dtype=wp.float64), ng: int, limiter: int):
    x, y = wp.tid()
    i = x + ng - 2
    j = y + ng - 2
    dx = wp.float64(0.5) * (a[i + 1, j] - a[i - 1, j])
    dy = wp.float64(0.5) * (a[i, j + 1] - a[i, j - 1])
    if limiter != 0:
        dx = mc(a[i + 1, j] - a[i, j], a[i, j] - a[i - 1, j], dx)
        dy = mc(a[i, j + 1] - a[i, j], a[i, j] - a[i, j - 1], dy)
    sx[i, j] = dx
    sy[i, j] = dy


@wp.kernel
def fourth_order(a: wp.array2d(dtype=wp.float64), tx: wp.array2d(dtype=wp.float64),
                 ty: wp.array2d(dtype=wp.float64), sx: wp.array2d(dtype=wp.float64),
                 sy: wp.array2d(dtype=wp.float64), ng: int):
    x, y = wp.tid()
    i = x + ng - 2
    j = y + ng - 2
    dx = (wp.float64(2.0) / wp.float64(3.0)) * (a[i + 1, j] - a[i - 1, j] - wp.float64(0.25) * (tx[i + 1, j] + tx[i - 1, j]))
    dy = (wp.float64(2.0) / wp.float64(3.0)) * (a[i, j + 1] - a[i, j - 1] - wp.float64(0.25) * (ty[i, j + 1] + ty[i, j - 1]))
    sx[i, j] = mc(a[i + 1, j] - a[i, j], a[i, j] - a[i - 1, j], dx)
    sy[i, j] = mc(a[i, j + 1] - a[i, j], a[i, j] - a[i, j - 1], dy)


@wp.kernel
def interfaces(a: wp.array2d(dtype=wp.float64), sx: wp.array2d(dtype=wp.float64),
               sy: wp.array2d(dtype=wp.float64), ax: wp.array2d(dtype=wp.float64),
               ay: wp.array2d(dtype=wp.float64), ng: int, u: wp.float64,
               v: wp.float64, cx: wp.float64, cy: wp.float64):
    x, y = wp.tid()
    i = x + ng - 1
    j = y + ng - 1
    if u < wp.float64(0.0):
        ax[i, j] = a[i, j] - wp.float64(0.5) * (wp.float64(1.0) + cx) * sx[i, j]
    else:
        ax[i, j] = a[i - 1, j] + wp.float64(0.5) * (wp.float64(1.0) - cx) * sx[i - 1, j]
    if v < wp.float64(0.0):
        ay[i, j] = a[i, j] - wp.float64(0.5) * (wp.float64(1.0) + cy) * sy[i, j]
    else:
        ay[i, j] = a[i, j - 1] + wp.float64(0.5) * (wp.float64(1.0) - cy) * sy[i, j - 1]


@wp.kernel
def fluxes(ax: wp.array2d(dtype=wp.float64), ay: wp.array2d(dtype=wp.float64),
           fx: wp.array2d(dtype=wp.float64), fy: wp.array2d(dtype=wp.float64),
           ng: int, u: wp.float64, v: wp.float64, hx: wp.float64, hy: wp.float64):
    x, y = wp.tid()
    i = x + ng - 1
    j = y + ng - 1
    mx = int(0)
    my = int(0)
    if u > wp.float64(0.0):
        mx = -1
    if v > wp.float64(0.0):
        my = -1
    fx[i, j] = u * (ax[i, j] - hy * (v * ay[i + mx, j + 1] - v * ay[i + mx, j]))
    fy[i, j] = v * (ay[i, j] - hx * (u * ax[i + 1, j + my] - u * ax[i, j + my]))


@wp.kernel
def update(a: wp.array2d(dtype=wp.float64), fx: wp.array2d(dtype=wp.float64),
           fy: wp.array2d(dtype=wp.float64), ng: int, dtdx: wp.float64, dtdy: wp.float64):
    x, y = wp.tid()
    i = x + ng
    j = y + ng
    a[i, j] = a[i, j] + dtdx * (fx[i, j] - fx[i + 1, j]) + dtdy * (fy[i, j] - fy[i, j + 1])


class Advection:
    """Device-resident state and scratch arrays; no host transfers during step().

    Input/output use pyro's (x, y) ordering with four ghost cells per side.
    Call numpy() explicitly to synchronize and retrieve the evolved state.
    """
    def __init__(self, density, *, nx, ny, dx, dy, u=1.0, v=1.0, limiter=2, device="cpu",
                 boundaries=("periodic", "periodic", "periodic", "periodic")):
        if limiter not in (0, 1, 2):
            raise ValueError("supported limiters: 0, 1, 2")
        if not isinstance(nx, int) or not isinstance(ny, int) or min(nx, ny) < 4:
            raise ValueError("nx and ny must be integers >= 4")
        if not all(math.isfinite(x) for x in (dx, dy, u, v)) or min(dx, dy) <= 0:
            raise ValueError("positive finite spacing and finite velocities required")
        if len(boundaries) != 4 or any(b not in ("periodic", "outflow") for b in boundaries):
            raise ValueError("four periodic or outflow boundaries required (xl, xr, yl, yr)")
        for lo, hi in (boundaries[:2], boundaries[2:]):
            if (lo == "periodic") != (hi == "periodic"):
                raise ValueError("periodic boundaries must be paired in each direction")
        self.boundaries = tuple(boundaries)
        self.ng = 4
        self.nx, self.ny = nx, ny
        self.dx, self.dy, self.u, self.v = dx, dy, u, v
        self.limiter = limiter
        self.device = wp.get_device(device)
        shape = (nx + 8, ny + 8)
        density = np.asarray(density, dtype=np.float64)
        if density.shape != shape or not np.isfinite(density).all():
            raise ValueError(f"density must be finite and have shape {shape}")
        self.a = wp.array(np.ascontiguousarray(density), dtype=wp.float64, device=self.device)
        self.sx, self.sy, self.tx, self.ty, self.ax, self.ay, self.fx, self.fy = (
            wp.zeros(shape, dtype=wp.float64, device=self.device) for _ in range(8))

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def fill_boundary(self):
        """Fill ghosts without transferring the density to the host."""
        self._launch(fill_ghosts, self.a.shape, [self.a, self.nx, self.ny, self.ng,
                     int(self.boundaries[0] == "periodic"), int(self.boundaries[2] == "periodic")])

    def prepare(self, dt):
        """Fill boundary ghosts and construct slopes, states, and CTU fluxes."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if max(abs(self.u) * dt / self.dx, abs(self.v) * dt / self.dy) > 1 + 1e-14:
            raise ValueError("advective CFL exceeds 1")
        f = wp.float64
        self.fill_boundary()
        target_x, target_y = (self.tx, self.ty) if self.limiter == 2 else (self.sx, self.sy)
        self._launch(slopes, (self.nx + 4, self.ny + 4), [self.a, target_x, target_y, self.ng, self.limiter])
        if self.limiter == 2:
            self._launch(fourth_order, (self.nx + 4, self.ny + 4), [self.a, self.tx, self.ty, self.sx, self.sy, self.ng])
        self._launch(interfaces, (self.nx + 2, self.ny + 2), [self.a, self.sx, self.sy, self.ax, self.ay, self.ng,
                     f(self.u), f(self.v), f(self.u * dt / self.dx), f(self.v * dt / self.dy)])
        self._launch(fluxes, (self.nx + 2, self.ny + 2), [self.ax, self.ay, self.fx, self.fy, self.ng,
                     f(self.u), f(self.v), f(0.5 * dt / self.dx), f(0.5 * dt / self.dy)])

    def step(self, dt):
        self.prepare(dt)
        self._launch(update, (self.nx, self.ny), [self.a, self.fx, self.fy, self.ng,
                     wp.float64(dt / self.dx), wp.float64(dt / self.dy)])

    def upload(self, density):
        """Load the authoritative host state before a driver-managed step."""
        host = wp.array(np.ascontiguousarray(density, dtype=np.float64),
                        dtype=wp.float64, device="cpu", copy=False)
        wp.copy(self.a, host)
        # Keep a possibly temporary contiguous buffer alive until the copy ends.
        wp.synchronize_device(self.device)

    def numpy(self):
        return self.a.numpy()
