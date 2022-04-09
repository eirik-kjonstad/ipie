import numpy
import pytest
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.estimators.generic import (
        local_energy_generic_opt,
        local_energy_generic_cholesky_opt,
        )
from ipie.legacy.estimators.generic import (
        local_energy_generic_cholesky_opt_batched,
        local_energy_generic_cholesky,
        )
from ipie.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd
        )

# FDM Implement half rotated integrals
# @pytest.mark.unit
def test_local_energy_opt():
    numpy.random.seed(7)
    nmo = 24
    nelec = (4,2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec, h1e=numpy.array([h1e, h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc, inputs={'integral_tensor': True})
    wfn = get_random_nomsd(sys, ndet=1, cplx=False)
    trial = MultiSlater(sys, wfn)
    trial.half_rotate(sys)
    e = local_energy_generic_opt(sys, trial.G, trial.Ghalf, trial._rchol)
    assert e[0] == pytest.approx(20.6826247016273)
    assert e[1] == pytest.approx(23.0173528796140)
    assert e[2] == pytest.approx(-2.3347281779866)

@pytest.mark.unit
def test_local_energy_cholesky():
    numpy.random.seed(7)
    nmo = 24
    nelec = (4,2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=1, cplx=False)
    trial = MultiSlater(system, ham, wfn)
    e = local_energy_generic_cholesky(system, ham, trial.G, Ghalf=trial.Ghalf)
    assert e[0] == pytest.approx(20.6826247016273)
    assert e[1] == pytest.approx(23.0173528796140)
    assert e[2] == pytest.approx(-2.3347281779866)

@pytest.mark.unit
def test_local_energy_cholesky_opt():
    numpy.random.seed(7)
    nmo = 24
    nelec = (4,2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec) 
    ham = HamGeneric (h1e=numpy.array([h1e, h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=1, cplx=False)
    trial = MultiSlater(system, ham, wfn)
    trial.half_rotate(system, ham)
    e = local_energy_generic_cholesky_opt(system, ham.ecore, trial.Ghalf[0],trial.Ghalf[1], trial._rH1a, trial._rH1b, trial._rchola, trial._rcholb)
    assert e[0] == pytest.approx(20.6826247016273)
    assert e[1] == pytest.approx(23.0173528796140)
    assert e[2] == pytest.approx(-2.3347281779866)

def test_local_energy_cholesky_opt_batched():
    numpy.random.seed(7)
    nmo = 24
    nelec = (4,2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric (h1e=numpy.array([h1e, h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=1, cplx=False)
    trial = MultiSlater(system, ham, wfn)
    trial.half_rotate(system, ham)
    Ga_batched = numpy.array([trial.G[0], trial.G[0], trial.G[0]])
    Gb_batched = numpy.array([trial.G[1], trial.G[1], trial.G[1]])
    Gahalf_batched = numpy.array([trial.Ghalf[0], trial.Ghalf[0], trial.Ghalf[0]])
    Gbhalf_batched = numpy.array([trial.Ghalf[1], trial.Ghalf[1], trial.Ghalf[1]])
    res = local_energy_generic_cholesky_opt_batched(system, ham, Ga_batched, Gb_batched,
                                                  Gahalf_batched, Gbhalf_batched,
                                                  trial._rchola, trial._rcholb)
    assert len(res[0]) == 3
    assert len(res[1]) == 3
    assert len(res[2]) == 3
    assert numpy.allclose(res[0], 20.6826247016273)
    assert numpy.allclose(res[1], 23.0173528796140)
    assert numpy.allclose(res[2], -2.3347281779866)

if __name__ == '__main__':
    test_local_energy_cholesky()
    test_local_energy_cholesky_opt()