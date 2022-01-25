from __future__ import print_function
import unittest
import numpy as np
import copy
from baseclasses import AeroProblem
import sys
from pprint import pprint as pp
import os

from numpy.lib.financial import ipmt

baseDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(baseDir, "../../"))
from adflow.pyADflow import ADFLOW
from mpi4py import MPI


class BaseHeatXferTest:
    "class to hold the common setup method for the tests"

    def setUp(self):
        gridFile = os.path.join(baseDir, "../input_files/nacelle_L5_1e-3.cgns")
        # gridFile = os.path.join(baseDir, "../input_files/nacelle_L5_1e-5.cgns")

        self.aero_options = {
            # I/O Parameters
            "gridFile": gridFile,
            # "restartFile": gridFile,
            "outputDirectory": "../output_files",
            "monitorvariables": ["cpu", "totheattransfer"],
            "writeTecplotSurfaceSolution": False,
            "writevolumesolution": True,
            "writesurfacesolution": True,
            "liftindex": 3,
            # Physics Parameters
            "equationType": "RANS",
            # Solver Parameters
            "smoother": "Runge-Kutta",
            "CFL": 1.5,
            "MGCycle": "sg",
            "MGStartLevel": -1,
            # ANK Solver Parameters
            "useANKSolver": True,
            "nsubiterturb": 5,
            "anksecondordswitchtol": 1e-2,
            "ankinnerpreconits": 2,
            "ankouterpreconits": 1,
            "anklinresmax": 0.1,
            "infchangecorrection": True,
            "ankcfllimit": 1e4,
            # NK Solver Parameters
            "useNKSolver": True,
            "nkswitchtol": 1e-6,
            "NKSubspaceSize": 120,
            # Termination Criteria
            "L2Convergence": 1e-12,
            "nCycles": 1000,
            # "RKReset": True,
            # "nRKReset": 20,
            # "nCycles": 1,
            "adjointl2convergence": 1e-15,
            "adjointSubspaceSize": 500,
            "heatxferratesAsFluxes": False,
            "ADPC": True,
            "setMonitor": True,
            # "partitionLikeNProc": 50,
            # "discretization" : "upwind",
        }

        self.ap = AeroProblem(
            name="fc_runup",
            V=32,  # m/s
            T=273 + 60,  # kelvin
            P=93e3,  # pa
            areaRef=0.1615 ** 2 * np.pi / 4,  # m^2 0.1615 is diameter of motor
            chordRef=0.9144,  # m
            evalFuncs=["cd", "totheattransfer", "havg", "hot_area", "cdv", "cdp"],
            alpha=0.0,
            beta=0.00,
            xRef=0.0,
            yRef=0.0,
            zRef=0.0,
        )

        self.group = "isothermalwall"
        self.BCVar = "Temperature"
        self.BCDV = "wall_temp"
        self.ap.setBCVar(self.BCVar, 400.0, self.group)
        self.ap.addDV(self.BCVar, name=self.BCDV, familyGroup=self.group)

        # these values come from inspecting the cgns mesh itself
        self.CFDSolver = ADFLOW(options=self.aero_options)
        self.CFDSolver.addFunction("area", self.group, name="hot_area")

        self.CFDSolver.getResidual(self.ap)
        # self.CFDSolver(self.ap)


class HeatXferTests(BaseHeatXferTest, unittest.TestCase):
    def test_bcdata(self):
        "Check that the temperature data set on the surface is read correctly"
        bc_data = self.CFDSolver.getBCData([self.group])
        for patch in bc_data:
            for arr in patch:
                np.testing.assert_array_equal(arr.data, np.ones(arr.size) * 400.0)

    def test_heatxfer(self):
        "Check that the func totheattransfer is calculated right"

        self.CFDSolver(self.ap)

        funcs = {}
        self.CFDSolver.evalFunctions(self.ap, funcs)

        Q_dot = funcs[self.ap.name + "_totheattransfer"]
        q = self.CFDSolver.getHeatXferRates(groupName=self.group)

        np.testing.assert_allclose(Q_dot, np.sum(q), rtol=1e-15)

    # def test_heatfluxes(self):
    #     "Check that the heat fluxes are calculated right"
    #     self.CFDSolver(self.ap)

    #     funcs = {}
    #     self.CFDSolver.evalFunctions(self.ap, funcs)

    #     q = self.CFDSolver.getHeatFluxes(groupName=self.group)

    #     # this one is a regression test, becuase I could think of something to compare with
    #     np.testing.assert_allclose(-191858.788552773, np.sum(q), rtol=5e-14)


# region derivative tests


class HeatXferAeroDVDerivsTests(BaseHeatXferTest, unittest.TestCase):
    def test_fwd_AeroDV(self):
        "test the FWD derivatives of the functional wrt alpha"

        self.ap.addDV("alpha", value=0.0)
        self.ap.DVs.pop(self.BCDV)

        xDvDot = {"alpha_" + self.ap.name: 1.0}

        resDot_FD, funcsDot_FD, fDot_FD, hfDot_FD = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="FD", h=1e-10
        )

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )
        np.testing.assert_allclose(resDot_FD, resDot, atol=10e-6)

        for func in funcsDot:
            print(func, funcsDot_FD[func], funcsDot[func])
            np.testing.assert_allclose(funcsDot_FD[func], funcsDot[func], err_msg=func, atol=5e-8)

        # we know what the correct answer should be so use that instead of the FD value
        np.testing.assert_allclose(fDot * 0.0, fDot, atol=1e-7)

        np.testing.assert_allclose(hfDot_FD * 0.0, hfDot, atol=1e-7)

    def cmplx_test_fwd_AeroDV(self):
        "test the FWD derivatives of the functional wrt Alpha using CS"
        self.ap.addDV("alpha", value=0.0)
        self.ap.DVs.pop(self.BCDV)

        from adflow.pyADflow_C import ADFLOW_C

        self.CFDSolver = ADFLOW_C(options=self.aero_options)
        self.CFDSolver.getResidual(self.ap)

        xDvDot = {"alpha_" + self.ap.name: 1.0}

        resDot_CS, funcsDot_CS, fDot_CS, hfDot_CS = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="CS", h=1e-200
        )

        for func in funcsDot_CS:
            print(func, funcsDot_CS[func])

    def test_bwd_AeroDV(self):
        "test the BWD derivatives of the functional"

        self.ap.addDV("alpha", value=0.0)
        self.ap.DVs.pop(self.BCDV)

        xDvDot = {"alpha_" + self.ap.name: 1.0}

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        dwBar = self.CFDSolver.getStatePerturbation(314)
        fBar = self.CFDSolver.getSurfacePerturbation(314)
        hxferBar = fBar[:, 0]

        funcsBar = {}
        for key in self.ap.evalFuncs:
            funcsBar[key] = 1.0

        xVBar = self.CFDSolver.computeJacobianVectorProductBwd(
            resBar=dwBar, fBar=fBar, hfBar=hxferBar, funcsBar=funcsBar, xDvDeriv=True
        )

        # we have to add up all the parts
        _sum = 0
        _sum += np.dot(resDot, dwBar)
        _sum += np.dot(fDot.flatten(), fBar.flatten())
        _sum += np.dot(hfDot.flatten(), hxferBar.flatten())
        _sum += np.sum([x for x in funcsDot.values()])

        for dv in xDvDot:
            np.testing.assert_array_almost_equal(np.dot(xDvDot[dv], xVBar[dv]), _sum, decimal=14)


class HeatXferBCDVDerivsTests(BaseHeatXferTest, unittest.TestCase):
    def test_fwd_BCDV(self):
        "test the FWD derivatives of the functional wrt BC DVs"

        xDvDot = {self.BCDV: 1.0}

        resDot_FD, funcsDot_FD, fDot_FD, hfDot_FD = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="FD", h=1e-6
        )

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        np.testing.assert_allclose(resDot_FD, resDot, atol=1e-6)

        for func in funcsDot:
            np.testing.assert_allclose(funcsDot_FD[func], funcsDot[func], err_msg=func, atol=1e-5)

        # we know what the correct answer should be so use that instead of the FD value
        np.testing.assert_allclose(fDot * 0.0, fDot, atol=1e-10)

        np.testing.assert_allclose(hfDot_FD, hfDot, atol=1e-10)

    def cmplxtest_fwd_BCDV(self):
        "test the FWD derivatives of the functional"

        from adflow.pyADflow_C import ADFLOW_C

        self.CFDSolver = ADFLOW_C(options=self.aero_options)

        self.ap.setBCVar(self.BCVar, 300.0, self.group)
        self.ap.addDV(self.BCVar, name=self.BCDV, familyGroup=self.group)
        self.CFDSoler.getResidual(self.ap)

        xDvDot = {self.BCDV: 1.0}

        resDot_CS, funcsDot_CS, fDot_CS, hfDot_CS = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="CS", h=1e-200
        )

    def test_bwd_BCDV(self):
        "test the FWD derivatives of the functional"

        func = self.BCDV
        xDvDot = {func: 1.0}

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        dwBar = self.CFDSolver.getStatePerturbation(314)
        fBar = self.CFDSolver.getSurfacePerturbation(314)
        hxferBar = fBar[:, 0]

        funcsBar = {}
        for key in self.ap.evalFuncs:
            funcsBar[key] = 1.0

        xVBar = self.CFDSolver.computeJacobianVectorProductBwd(
            resBar=dwBar, fBar=fBar, hfBar=hxferBar, funcsBar=funcsBar, xDvDeriv=True
        )

        # we have to add up all the parts
        _sum = 0
        _sum += np.dot(resDot, dwBar)
        _sum += np.dot(fDot.flatten(), fBar.flatten())
        _sum += np.dot(hfDot.flatten(), hxferBar.flatten())
        _sum += np.sum([x for x in funcsDot.values()])

        np.testing.assert_array_almost_equal(np.dot(xDvDot[func], xVBar[func]), _sum, decimal=14)


class HeatXferXVDerivsTests(BaseHeatXferTest, unittest.TestCase):
    def test_fwd_XV(self):
        "test the FWD derivatives wrt to nodal displacements"

        xVDot = self.CFDSolver.getSpatialPerturbation(321)

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            xVDot=xVDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        resDot_FD, funcsDot_FD, fDot_FD, hfDot_FD = self.CFDSolver.computeJacobianVectorProductFwd(
            xVDot=xVDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="FD", h=1e-8
        )

        # these are checked against CS because the dervatives appear to be poorly conditioned
        # import pickle
        # with open("resDot_CS.p", "rb") as f:
        # resDot_CS = pickle.load(f)

        # np.testing.assert_allclose(resDot_CS, resDot, rtol=5e-8)

        for func in funcsDot:
            np.testing.assert_allclose(funcsDot_FD[func], funcsDot[func], err_msg=func, rtol=8e-5)

        np.testing.assert_allclose(fDot_FD, fDot, rtol=2.5e-4)

        np.testing.assert_allclose(hfDot_FD[hfDot_FD != 0], hfDot[hfDot_FD != 0], rtol=5e-5)

    def test_bwd_XV(self):
        "test the FWD derivatives of the functional"

        xVDot = self.CFDSolver.getSpatialPerturbation(321)

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            xVDot=xVDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        dwBar = self.CFDSolver.getStatePerturbation(314)
        fBar = self.CFDSolver.getSurfacePerturbation(314)
        hxferBar = fBar[:, 0]

        funcsBar = {}
        for key in self.ap.evalFuncs:
            funcsBar[key] = 1.0

        xVBar = self.CFDSolver.computeJacobianVectorProductBwd(
            resBar=dwBar, fBar=fBar, hfBar=hxferBar, funcsBar=funcsBar, xVDeriv=True
        )

        # we have to add up all the parts
        _sum = 0
        _sum += np.dot(resDot, dwBar)
        _sum += np.dot(fDot.flatten(), fBar.flatten())
        _sum += np.dot(hfDot.flatten(), hxferBar.flatten())
        _sum += np.sum([x for x in funcsDot.values()])

        np.testing.assert_allclose(np.dot(xVDot, xVBar), _sum, rtol=5e-14)


class HeatXferWDerivsTests(BaseHeatXferTest, unittest.TestCase):
    def test_fwd_W(self):
        "test the FWD derivatives wrt to the states"

        # xVDot = self.CFDSolver.getSpatialPerturbation(321)
        wDot = self.CFDSolver.getStatePerturbation(314)

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            wDot=wDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        resDot_FD, funcsDot_FD, fDot_FD, hfDot_FD = self.CFDSolver.computeJacobianVectorProductFwd(
            wDot=wDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="FD", h=1e-8
        )

        # TODO: re train the complex ref and check again
        # np.testing.assert_allclose(resDot_CS, resDot, rtol=1e-11)

        for func in funcsDot:
            np.testing.assert_allclose(funcsDot_FD[func], funcsDot[func], err_msg=func, rtol=1e-5)

        np.testing.assert_allclose(fDot_FD, fDot, rtol=5e-5)

        np.testing.assert_allclose(hfDot_FD[hfDot_FD != 0], hfDot[hfDot_FD != 0], rtol=5e-5)

    def cmplxtest_fwd_W(self):
        "test the FWD derivatives of the functional"
        from adflow.pyADflow_C import ADFLOW_C

        self.CFDSolver = ADFLOW_C(options=self.aero_options)
        self.CFDSolver.getResidual(self.ap)
        # self.CFDSolver(self.ap)

        wDot = self.CFDSolver.getStatePerturbation(314)

        # resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
        # xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode='AD')

        resDot_CS, funcsDot_CS, fDot_CS, hfDot_CS = self.CFDSolver.computeJacobianVectorProductFwd(
            wDot=wDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="CS", h=1e-200
        )
        print(resDot_CS[1944])

        import pickle

        with open("resDot_CS_heatxfer.p", "wb") as f:
            pickle.dump(resDot_CS, f)

    def test_bwd_W(self):
        "test the BWD derivatives wrt the states"

        wDot = self.CFDSolver.getStatePerturbation(314)

        resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
            wDot=wDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode="AD"
        )

        dwBar = self.CFDSolver.getStatePerturbation(314)
        fBar = self.CFDSolver.getSurfacePerturbation(314)
        hxferBar = fBar[:, 0]

        funcsBar = {}
        for key in self.ap.evalFuncs:
            funcsBar[key] = 1.0

        wBar = self.CFDSolver.computeJacobianVectorProductBwd(
            resBar=dwBar, fBar=fBar, hfBar=hxferBar, funcsBar=funcsBar, wDeriv=True
        )

        # we have to add up all the parts
        _sum = 0
        _sum += np.dot(resDot, dwBar)
        _sum += np.dot(fDot.flatten(), fBar.flatten())
        _sum += np.dot(hfDot.flatten(), hxferBar.flatten())
        _sum += np.sum([x for x in funcsDot.values()])

        np.testing.assert_allclose(np.dot(wDot, wBar), _sum, rtol=6e-14)


# endregion


class TotalDerivsTests(BaseHeatXferTest, unittest.TestCase):
    def test_temp(self):
        "test the total derivatives wrt to temperature"

        funcsSens = {}
        funcs_base = {}
        funcs2 = {}
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs_base)
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSens)

        # self.ap.setBCVar(self.BCVar, 400.0 + h, self.group)

        # pertub the bc dv

        h = 1e-3
        self.ap.setDesignVars({self.BCDV: 400.0 + h})

        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs2)

        for f, v in funcsSens.items():
            v_fd = (funcs2[f] - funcs_base[f]) / h
            print(f, v, v_fd)
            v_ad = v[self.BCDV]
            if v == 0.0:
                np.testing.assert_allclose(v_ad, v_fd, err_msg=f, atol=6e-5)
            else:
                np.testing.assert_allclose(v_ad, v_fd, err_msg=f, rtol=6e-5)

    def test_xs(self):
        """
        test the derivatives wrt to nodal displacements
        adflow assumes that the nodal dvs are from ffds, so we can't just use
        evalfunctionssens
        """
        from idwarp import USMesh

        funcs_base = {}
        funcs2 = {}
        objective = "totheattransfer"
        mesh = USMesh(options={"gridFile": self.aero_options["gridFile"]})
        self.CFDSolver.setMesh(mesh)
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs_base)
        self.CFDSolver.solveAdjoint(self.ap, objective)

        psi = -1 * self.CFDSolver.getAdjoint(objective)

        xSDeriv = self.CFDSolver.computeJacobianVectorProductBwd(
            resBar=psi, funcsBar=self.CFDSolver._getFuncsBar(objective), xSDeriv=True, xDvDeriv=False
        )
        # xDvBar, xSDeriv = self.computeJacobianVectorProductBwd(resBar=psi, funcsBar=se/lf._getFuncsBar(objective), xSDeriv=True, xDvDeriv=True)

        xs = self.CFDSolver.getSurfaceCoordinates()
        h = 1e-9

        idxs_pertub = [(10, 1), (20, 0), (100, 2)]
        for idx in idxs_pertub:
            xs[idx] += h
            self.CFDSolver.setSurfaceCoordinates(xs)
            self.CFDSolver(self.ap)
            self.CFDSolver.evalFunctions(self.ap, funcs2)
            xs[idx] -= h

            f = self.ap.name + "_" + objective
            v_fd = (funcs2[f] - funcs_base[f]) / h
            v_ad = xSDeriv[idx]
            np.testing.assert_allclose(v_ad, v_fd, err_msg=str(idx) + f, rtol=6.5e-2)

    def test_aeroDVs(self):
        "test the total derivatives wrt to aero DVs"

        self.ap.addDV("alpha", value=1.0)
        self.ap.DVs.pop(self.BCDV)

        funcsSens = {}
        funcs_base = {}
        funcs2 = {}
        # self.CFDSolver(self.ap)
        # self.CFDSolver.evalFunctions(self.ap, funcs_base)
        # self.CFDSolver.evalFunctionsSens(self.ap,funcsSens, evalFuncs=["totheattransfer"])

        # # h = 1e-2
        # # self.ap.alpha += h
        # # self.CFDSolver(self.ap)
        # # self.CFDSolver.evalFunctions(self.ap, funcs2)

        # # for f, v in funcsSens.items():
        # #     print(f, v,  (funcs2[f] - funcs_base[f])/h)

        # psi = self.CFDSolver.curAP.adflowData.adjoints["totheattransfer"].copy()
        # RHS = self.CFDSolver.curAP.adflowData.adjointRHS["totheattransfer"].copy()

        # LHS = self.CFDSolver.computeJacobianVectorProductBwd(resBar=psi, wDeriv=True)
        # err = np.linalg.norm(LHS - RHS)
        # print(err)
        # LHS = self.CFDSolver.computeJacobianVectorProductBwdFast(resBar=psi)
        # err = np.linalg.norm(LHS - RHS)
        # print(err)s
        dwBar = self.CFDSolver.getStatePerturbation(314)

        psi = np.ones_like(dwBar)
        # wbar = self.CFDSolver.computeJacobianVectorProductBwd(resBar=psi, wDeriv=True)
        tmp1 = self.CFDSolver.computeJacobianVectorProductBwdFast(resBar=psi)
        # tmp2 = self.CFDSolver.computeJacobianVectorProductBwdFast(resBar=psi)
        print(np.linalg.norm(tmp1))

        # print(np.linalg.norm(tmp1 - tmp2))
        # print(np.linalg.norm(tmp1 - wbar))
        # tmp1 = self.CFDSolver.computeJacobianVectorProductBwdFast(resBar=psi)
        # print(np.linalg.norm(tmp1 - wbar))

        # h = 3e-4
        # self.ap.alpha += h
        # self.CFDSolver(self.ap)
        # self.CFDSolver.evalFunctions(self.ap, funcs2)

        # for f, v in funcsSens.items():
        #     v_fd = (funcs2[f] - funcs_base[f]) / h
        #     print(f, v, v_fd)
        #     v_ad = v["alpha_" + self.ap.name]
        #     if v == 0.0:
        #         np.testing.assert_allclose(v_ad, v_fd, err_msg=f, atol=1e-2)
        #     else:
        #         np.testing.assert_allclose(v_ad, v_fd, err_msg=f, rtol=1e-2)

    def cmplxtest_aeroDVs(self):
        "test the total derivatives wrt to aero DVs"

        from adflow.pyADflow_C import ADFLOW_C

        self.CFDSolver = ADFLOW_C(options=self.aero_options)
        self.CFDSolver.addFunction("area", self.group, name="hot_area")

        self.ap.addDV("alpha", value=1.0)
        self.ap.DVs.pop(self.BCDV)

        funcs2 = {}

        self.ap.alpha += 1e-200 * 1j
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs2)

        for f, v in funcs2.items():
            print(f, np.imag(v) / 1e-200)

    def cmplxtest_temp(self):
        "test temperature derivatives with complex step"

        from adflow.pyADflow_C import ADFLOW_C

        # import ipdb; ipdb.set_trace()
        self.CFDSolver = ADFLOW_C(options=self.aero_options)

        self.CFDSolver.addFunction("area", self.group, name="hot_area")

        self.ap.setBCVar(self.BCVar, 400.0 + 1e-200j, self.group)
        self.ap.addDV(self.BCVar, name=self.BCDV, familyGroup=self.group)
        self.CFDSolver(self.ap)
        funcs = {}
        self.CFDSolver.evalFunctions(self.ap, funcs)

        for f, v in funcs.items():
            print(f, np.imag(v) / 1e-200)

    def cmplxtest_xs(self):
        "test temperature derivatives with complex step"

        from adflow.pyADflow_C import ADFLOW_C
        from idwarp import USMesh_C

        self.aero_options["useNKsolver"] = False
        self.aero_options["useANKSolver"] = False
        self.aero_options["nCycles"] = 1e5
        self.aero_options["useNKsolver"] = False

        # import ipdb; ipdb.set_trace()
        self.CFDSolver = ADFLOW_C(options=self.aero_options)
        mesh = USMesh_C(options={"gridFile": self.aero_options["gridfile"]})
        self.CFDSolver.setMesh(mesh)
        self.CFDSolver.addFunction("area", self.group, name="hot_area")

        # self.ap.setBCVar(self.BCVar, 400.0+1e-200j, self.group)
        # self.ap.addDV(self.BCVar, name=self.BCDV, familyGroup=self.group)
        # for ii in
        xs = self.CFDSolver.getSurfaceCoordinates()
        xs[0, 0] += 1e-200j
        self.CFDSolver.setSurfaceCoordinates(xs)
        self.CFDSolver(self.ap)
        funcs = {}
        self.CFDSolver.evalFunctions(self.ap, funcs)

        for f, v in funcs.items():
            print(f, np.imag(v) / 1e-200)

        # xDvDot = {self.BCDV: 1.0}

        # resDot, funcsDot, fDot, hfDot = self.CFDSolver.computeJacobianVectorProductFwd(
        # xDvDot=xDvDot, residualDeriv=True, funcDeriv=True, fDeriv=True, hfDeriv=True, mode='AD')


if __name__ == "__main__":
    unittest.main()
