import tempfile
import unittest
from pathlib import Path

from dara.refine import RefinementPhase, do_refinement
from dara.result import parse_lst
from dara.search.phase_grouping import calculate_fom_and_strain


MINIMAL_LST = """Rietveld refinement to file(s) test.xy
BGMN version 4.2.23, 100 measured points, 5 peaks, 3 parameters
Start: Mon Jan 1 00:00:00 2024; End: Mon Jan 1 00:00:01 2024
5 iteration steps

Rp=4.14%  Rpb=50.39%  R=13.55%  Rwp=8.98% Rexp=1.47%
{extra}
Global parameters and GOALs
****************************
EPS2=-0.001657+-0.000033

Local parameters and GOALs for phase TestPhase
******************************************************
SpacegroupNo=225
HermannMauguin=F4/m-32/m
XrayDensity=6.760
Rphase=11.31%
UNIT=NM
A=0.418697+-0.000027
k1=0
B1=0.00798+-0.00022
GEWICHT=0.3827+-0.0049
GrainSize(1,1,1)=53.2+-1.5
Atomic positions for phase TestPhase
---------------------------------------------
  4     0.0000  0.0000  0.0000     E=(NI+2(1.0000))
  4     0.5000  0.5000  0.5000     E=(O-2(1.0000))
"""


class TestParseLstDurbinWatsonRho(unittest.TestCase):
    """Regression tests: parse_lst used to raise a pydantic ValidationError
    whenever the Durbin-Watson d / 1-rho lines were missing from the .lst
    file (the extraction fell back to None, but LstResult declared both as
    required floats), and the regexes only matched unsigned numbers, so a
    negative value hit the same failure."""

    def _write_and_parse(self, extra: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            lst_path = Path(tmpdir) / "test.lst"
            lst_path.write_text(MINIMAL_LST.format(extra=extra))
            return parse_lst(lst_path, phase_names=["TestPhase"])

    def test_missing_durbin_watson_and_rho_does_not_raise(self):
        result = self._write_and_parse("")
        self.assertIsNone(result.d)
        self.assertIsNone(result.rho)

    def test_negative_durbin_watson_and_rho_parse_correctly(self):
        result = self._write_and_parse("Durbin-Watson d=-0.5\n1-rho=-2.3%\n")
        self.assertEqual(result.d, -0.5)
        self.assertEqual(result.rho, -2.3)

    def test_positive_durbin_watson_and_rho_still_parse_correctly(self):
        result = self._write_and_parse("Durbin-Watson d=0.06\n1-rho=13.6%\n")
        self.assertEqual(result.d, 0.06)
        self.assertEqual(result.rho, 13.6)


class TestCalculateFomAndStrainHandlesMissingRho(unittest.TestCase):
    """calculate_fom_and_strain divides by result.lst_data.rho; confirm it
    falls back to the documented (0, 0, is_ordered) "cannot be calculated"
    result instead of raising when rho is None, consistent with how it
    already handles a missing refined lattice parameter."""

    def setUp(self):
        self.cif_paths = list((Path(__file__).parent / "test_data").glob("*.cif"))
        self.pattern_path = Path(__file__).parent / "test_data" / "BiFeO3.xy"

    def test_none_rho_returns_zero_fom_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            result = do_refinement(
                self.pattern_path,
                self.cif_paths,
                instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
                working_dir=tmpdir,
            )
            phase = RefinementPhase(path=self.cif_paths[0])

            result.lst_data.rho = None
            fom, lattice_strain, is_ordered = calculate_fom_and_strain(phase, result)
            self.assertEqual(fom, 0)
            self.assertEqual(lattice_strain, 0)


if __name__ == "__main__":
    unittest.main()
