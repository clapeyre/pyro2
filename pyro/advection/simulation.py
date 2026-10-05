import matplotlib.pyplot as plt
import numpy as np

import pyro.advection.advective_fluxes as flx
from pyro.advection.interface import linear_interface
from pyro.mesh import patch
from pyro.particles import particles
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup
from pyro.util import plot_tools


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

        self._warp = None
        if self._backend_name() == "warp":
            if self.rp.get_param("particles.do_particles"):
                raise ValueError("Warp advection does not support particles")
            if self.rp.get_param("mesh.grid_type") != "Cartesian2d":
                raise ValueError("Warp advection requires a Cartesian mesh")
            sides = ("xlboundary", "xrboundary", "ylboundary", "yrboundary")
            boundaries = tuple(self.rp.get_param("mesh." + side) for side in sides)
            self._validate_warp_boundaries(boundaries)

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = patch.CellCenterData2d(my_grid)
        bc = bc_setup(self.rp)[0]
        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # now set the initial conditions for the problem
        self.problem_func(self.cc_data, self.rp)
        self._advection_backend()

    def _backend_name(self):
        try:
            backend = self.rp.get_param("advection.backend")
        except KeyError:
            backend = "numpy"
        if backend not in ("numpy", "warp"):
            raise ValueError("advection.backend must be numpy or warp")
        if backend == "warp" and self.solver_name != "advection":
            raise ValueError("Warp is supported only by the advection solver")
        return backend

    @staticmethod
    def _validate_warp_boundaries(boundaries):
        if any(b not in ("periodic", "outflow") for b in boundaries):
            raise ValueError("Warp advection supports only periodic or outflow boundaries")
        for lo, hi in (boundaries[:2], boundaries[2:]):
            if (lo == "periodic") != (hi == "periodic"):
                raise ValueError("periodic boundaries must be paired in each direction")

    def _advection_backend(self):
        if self._backend_name() == "numpy":
            return None
        if self.rp.get_param("particles.do_particles"):
            raise ValueError("Warp advection does not support particles")
        grid = self.cc_data.grid
        if not isinstance(grid, patch.Cartesian2d) or grid.ng != 4:
            raise ValueError("Warp advection requires a Cartesian mesh with four ghost cells")
        bc = self.cc_data.BCs["density"]
        boundaries = tuple(getattr(bc, side) for side in ("xlb", "xrb", "ylb", "yrb"))
        self._validate_warp_boundaries(boundaries)
        if any(getattr(bc, side + "_value") is not None for side in ("xl", "xr", "yl", "yr")):
            raise ValueError("Warp advection does not support boundary value callbacks")
        try:
            device = self.rp.get_param("advection.warp_device")
        except KeyError:
            device = "cuda:0"
        u = self.rp.get_param("advection.u")
        v = self.rp.get_param("advection.v")
        limiter = self.rp.get_param("advection.limiter")
        key = (id(grid), grid.nx, grid.ny, grid.dx, grid.dy, u, v, limiter, device, boundaries)
        cached = getattr(self, "_warp", None)
        if cached is None or getattr(self, "_warp_key", None) != key:
            try:
                # Import only when the optional backend is selected.
                from pyro.advection.warp_backend import \
                    Advection  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as exc:
                if exc.name == "warp":
                    raise ImportError("Warp advection requires the optional dependency: pip install '.[warp]'") from exc
                raise
            self._warp = Advection(self.cc_data.get_var("density"),
                nx=grid.nx, ny=grid.ny, dx=grid.dx, dy=grid.dy,
                u=u, v=v, limiter=limiter, device=device, boundaries=boundaries)
            self._warp_key = key
        return self._warp

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.rp.get_param("advection.u")
        v = self.rp.get_param("advection.v")

        # the timestep is min(dx/|u|, dy/|v|)
        xtmp = self.cc_data.grid.dx/max(abs(u), self.SMALL)
        ytmp = self.cc_data.grid.dy/max(abs(v), self.SMALL)

        self.dt = cfl*min(xtmp, ytmp)

    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        warp_backend = self._advection_backend()
        if warp_backend is not None:
            density = self.cc_data.get_var("density")
            warp_backend.upload(density)
            warp_backend.step(self.dt)
            density[:, :] = warp_backend.numpy()
            self.cc_data.t += self.dt
            self.n += 1
            return

        dtdx = self.dt/self.cc_data.grid.dx
        dtdy = self.dt/self.cc_data.grid.dy

        flux_x, flux_y = flx.unsplit_fluxes(self.cc_data, self.rp, self.dt, "density", linear_interface)

        """
        do the differencing for the fluxes now.  Here, we use slices so we
        avoid slow loops in python.  This is equivalent to:

        myPatch.data[i,j] = myPatch.data[i,j] + \
                               dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                               dtdy*(flux_y[i,j] - flux_y[i,j+1])
        """

        dens = self.cc_data.get_var("density")

        dens.v()[:, :] = dens.v() + dtdx*(flux_x.v() - flux_x.ip(1)) + \
                                    dtdy*(flux_y.v() - flux_y.jp(1))

        if self.particles is not None:
            myg = self.cc_data.grid
            u = self.rp.get_param("advection.u")
            v = self.rp.get_param("advection.v")

            u2d = myg.scratch_array() + u
            v2d = myg.scratch_array() + v

            self.particles.update_particles(self.dt, u2d, v2d)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()

        dens = self.cc_data.get_var("density")

        myg = self.cc_data.grid

        _, axes, _ = plot_tools.setup_axes(myg, 1)

        # plot density
        ax = axes[0]
        img = ax.imshow(np.transpose(dens.v()),
                   interpolation="nearest", origin="lower",
                   extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                   cmap=self.cm)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # needed for PDF rendering
        cb = axes.cbar_axes[0].colorbar(img)
        cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")

        plt.title("density")

        if self.particles is not None:
            particle_positions = self.particles.get_positions()

            # dye particles
            colors = self.particles.get_init_positions()[:, 0]

            # plot particles
            ax.scatter(particle_positions[:, 0],
                particle_positions[:, 1], c=colors, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5f}")

        plt.pause(0.001)
        plt.draw()
