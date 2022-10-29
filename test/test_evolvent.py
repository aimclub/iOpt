import unittest
import sys

sys.path.insert(0,"..")

from iOpt.evolvent.evolvent import Evolvent


class TestEvolvent(unittest.TestCase):
    def setUp(self):
        self.ev = Evolvent([-1],[1])

    def test_Preimages(self):
        y = [0]
        self.assertEqual(self.ev.GetPreimages(y), 0.5)

    def test_XtoYandBack(self):
        x1 = 0.5
        y  = self.ev.GetImage(x1)
        x2 = self.ev.GetInverseImage(y)
        self.assertEqual(x1, x2)

# Executing the tests in the above test case class
if __name__ == "__main__":
 unittest.main()