import time
from typing import Optional

import numpy
import plum

from ipie.config import CommType, config, MPI
from ipie.estimators.generic import half_rotated_cholesky_jk_uhf
from ipie.estimators.greens_function_single_det import (
    greens_function_single_det,
    greens_function_single_det_batch,
)
from ipie.estimators.utils import gab_spin
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.propagation.force_bias import (
    construct_force_bias_batch_single_det,
    construct_force_bias_batch_single_det_chunked,
)
from ipie.propagation.overlap import calc_overlap_single_det_uhf
from ipie.trial_wavefunction.half_rotate import half_rotate_generic, half_rotate_chunked
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers
from typing import Union


# class for UHF trial
class SingleDet(TrialWavefunctionBase):
    def __init__(self, wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 2
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")
        self.psi = wavefunction
        self.num_elec = num_elec
        self._num_dets = 1
        self._max_num_dets = 1
        imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
        if imag_norm <= 1e-8:
            # print("# making trial wavefunction MO coefficient real")
            self.psi = numpy.array(self.psi.real, dtype=numpy.float64)

        self.psi0a = self.psi[:, : self.nalpha]
        self.psi0b = self.psi[:, self.nalpha :]
        self.G, self.Ghalf = gab_spin(self.psi, self.psi, self.nalpha, self.nbeta)
        self.handler = handler

        self.psi0a = numpy.ascontiguousarray(self.psi0a)
        self.psi0b = numpy.ascontiguousarray(self.psi0b)

    def build(self) -> None:
        pass

    @property
    def num_dets(self) -> int:
        return 1

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        raise RuntimeError("Cannot modify number of determinants in SingleDet trial.")

    def calculate_energy(self, system, hamiltonian) -> numpy.ndarray:
        """
        `system` is just for backwards compatibillity.
        """
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        start = time.time()
        self.e1b = (
            numpy.sum(self.Ghalf[0] * self._rH1a)
            + numpy.sum(self.Ghalf[1] * self._rH1b)
            + hamiltonian.ecore
        )
        self.ej, self.ek = half_rotated_cholesky_jk_uhf(self, hamiltonian, self.Ghalf)
        self.e2b = self.ej - self.ek
        self.energy = self.e1b + self.e2b

        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )
            print(f"# Time to evaluate trial energy: {time.time() - start} s")

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericRealChol,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericRealCholChunked,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_chunked(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola_chunk = rot_chol[0][0]
        self._rcholb_chunk = rot_chol[1][0]
        self.half_rotated = True

        # rot_1body_1 = numpy.load('../Test_Disk_nochunk/rot_1body.npy')
        # rot_chol_1 = numpy.load('../Test_Disk_nochunk/rot_chol.npy')

        # print('compare', [numpy.allclose(rot_1body, rot_1body_1), numpy.allclose(rot_chol, rot_chol_1)])

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericComplexChol,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0][0]
        self._rcholb = rot_chol[1][0][0]
        self._rcholbara = rot_chol[0][1][0]
        self._rcholbarb = rot_chol[1][1][0]
        self._rAa = rot_chol[0][2][0]
        self._rAb = rot_chol[1][2][0]
        self._rBa = rot_chol[0][3][0]
        self._rBb = rot_chol[1][3][0]
        self.half_rotated = True

    def calc_overlap(self, walkers) -> numpy.ndarray:
        return calc_overlap_single_det_uhf(walkers, self)

    def calc_greens_function(self, walkers, build_full: bool = False) -> numpy.ndarray:
        if config.get_option("use_gpu"):
            return greens_function_single_det_batch(walkers, self, build_full=build_full)
        else:
            return greens_function_single_det(walkers, self, build_full=build_full)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: Union[GenericRealChol, GenericRealCholChunked],
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> xp.ndarray:
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walkers, self, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(
                hamiltonian, walkers, self._rchola, self._rcholb
            )

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: GenericComplexChol,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> numpy.ndarray:
        return construct_force_bias_batch_single_det(
            hamiltonian, walkers, self._rAa, self._rAb, self._rBa, self._rBb
        )
