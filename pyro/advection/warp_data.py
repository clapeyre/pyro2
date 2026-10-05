"""Advection-local host mirror for opt-in device-resident stepping.

Host array access synchronizes and makes the mirror authoritative for the next
step. Reacquire host views after stepping; retained views are not refreshed until
another host access. The default backend does not use this adapter.
"""
from pyro.mesh import patch


class ResidentData(patch.CellCenterData2d):
    def __init__(self, grid, *, dtype=None):
        self.backend = None
        self.device_dirty = False
        self.host_dirty = True
        self.ghosts_pending = False
        if dtype is None:
            super().__init__(grid)
        else:
            super().__init__(grid, dtype=dtype)

    @classmethod
    def from_data(cls, data):
        """Adopt loaded host data without changing its array or metadata."""
        result = cls(data.grid, dtype=data.dtype)
        result.__dict__.update({k: v for k, v in data.__dict__.items() if k != "data"})
        result.data = data.data
        return result

    @property
    def data(self):
        self.synchronize()
        # NumPy views are mutable: even a read may be followed by an edit.
        self.host_dirty = True
        return self._host_data

    @data.setter
    def data(self, value):
        # Replacing the array makes the new host state authoritative.
        self._host_data = value
        self.device_dirty = False
        self.host_dirty = True
        self.ghosts_pending = False

    def synchronize(self):
        """Refresh the mirror without exporting a mutable host view."""
        if self.device_dirty or self.ghosts_pending:
            if self.ghosts_pending:
                self.backend.fill_boundary()
            self._host_data[:, :, 0] = self.backend.numpy()
            self.device_dirty = False
            self.ghosts_pending = False

    def begin_step(self, backend):
        if self.backend is not backend:
            # The new backend was constructed from the synchronized mirror.
            self.backend = backend
            self.host_dirty = False
        elif self.host_dirty:
            backend.upload(self._host_data[:, :, 0])
            self.host_dirty = False

    def end_step(self):
        self.device_dirty = True
        self.ghosts_pending = False

    def write_data(self, f):
        # The built-in writer only reads the mirror. Output does not require
        # uploading that unchanged mirror again on the next step.
        was_dirty = self.host_dirty
        try:
            super().write_data(f)
        finally:
            self.host_dirty = was_dirty

    def detach(self):
        """Return to host stepping with current state and boundary cells."""
        self.synchronize()
        self.backend = None
        self.host_dirty = True
        super().fill_BC_all()

    def fill_BC(self, name):
        if self.backend is not None and not self.host_dirty:
            # prepare() fills ghosts before evolving. A subsequent host access
            # must also honor an explicit fill requested between steps.
            self.ghosts_pending = True
        else:
            super().fill_BC(name)
