# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

from typing import Optional

from ipie.addons.free_projection.propagation.CCSD import CCSD
from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import qr, qr_mode, synchronize
from ipie.walkers.uhf_walkers import UHFWalkers, UHFWalkersParticleHole


class UHFWalkersFP(UHFWalkers):
    """UHF style walker specialized for its use with free projection."""

    def initialize_walkers(self, ccsd: Optional[CCSD] = None):
        """Initialize walkers using CCSD."""
        if ccsd is not None:
            ccsd_walkers = ccsd.get_walkers(self.nwalkers)
            self.phia = ccsd_walkers.copy()
            self.phib = ccsd_walkers.copy()

    def set_walkers(self, walkers_a, walkers_b):
        assert walkers_a.shape == (self.nwalkers, self.nbasis, self.nup)
        assert walkers_b.shape == (self.nwalkers, self.nbasis, self.ndown)
        self.phia = walkers_a.copy()
        self.phib = walkers_b.copy()

    def orthogonalise(self, free_projection=False):
        """Orthogonalise all walkers.

        Parameters
        ----------
        free_projection : bool
            This flag is not used here.
        """
        #detR = self.reortho()
        detR = self.reortho_fromphaseless()
        magn, dtheta = xp.abs(self.detR), xp.angle(self.detR)
        self.weight *= magn
        self.phase *= xp.exp(1j * dtheta)
        return detR

    def reortho_batched(self):
        assert config.get_option("use_gpu")
        (self.phia, Rup) = qr(self.phia, mode=qr_mode)
        Rup_diag = xp.einsum("wii->wi", Rup)
        det = xp.prod(Rup_diag, axis=1)

        if self.ndown > 0:
            (self.phib, Rdn) = qr(self.phib, mode=qr_mode)
            Rdn_diag = xp.einsum("wii->wi", Rdn)
            det *= xp.prod(Rdn_diag, axis=1)
        self.detR = det
        self.ovlp = self.ovlp / self.detR
        synchronize()
        return self.detR

    def reortho_fromphaseless(self):
        """reorthogonalise walkers."""
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        ndown = self.ndown
        detR = []
        for iw in range(self.nwalkers):
            (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
            # TODO: FDM This isn't really necessary, the absolute value of the
            # weight is used for population control so this shouldn't matter.
            # I think this is a legacy thing.
            # Wanted detR factors to remain positive, dump the sign in orbitals.
            Rup_diag = xp.diag(Rup)
            signs_up = xp.sign(Rup_diag)
            self.phia[iw] = xp.dot(self.phia[iw], xp.diag(signs_up))

            # include overlap factor
            # det(R) = \prod_ii R_ii
            # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
            # C factor included to avoid over/underflow
            log_det = xp.sum(xp.log(xp.abs(Rup_diag)))

            if ndown > 0:
                (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                Rdn_diag = xp.diag(Rdn)
                signs_dn = xp.sign(Rdn_diag)
                self.phib[iw] = xp.dot(self.phib[iw], xp.diag(signs_dn))
                log_det += sum(xp.log(abs(Rdn_diag)))

            detR += [xp.exp(log_det - self.detR_shift[iw])]
            self.log_detR[iw] += xp.log(detR[iw])
            self.detR[iw] = detR[iw]
            self.ovlp[iw] = self.ovlp[iw] / detR[iw]

        synchronize()
        return detR

    def reortho(self):
        """reorthogonalise walkers for free projection, retaining normalization.

        parameters
        ----------
        """
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        else:
            ndown = self.ndown
            detR = []
            for iw in range(self.nwalkers):
                (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
                det_i = xp.prod(xp.diag(Rup))

                if ndown > 0:
                    (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                    det_i *= xp.prod(xp.diag(Rdn))

                detR += [det_i]
                self.log_detR[iw] += xp.log(detR[iw])
                self.detR[iw] = detR[iw]
                self.ovlp[iw] = self.ovlp[iw] / detR[iw]

        synchronize()
        return self.detR


class UHFWalkersParticleHoleFP(UHFWalkersFP, UHFWalkersParticleHole):
    """MSD walker for free projection."""
