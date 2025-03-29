import matplotlib
matplotlib.use('Agg')
from dfastbe.cmd import run


def test_bank_erosion():
    file = "erosion"
    language = "UK"
    config_file = f"tests/data/{file}/meuse_manual.cfg"
    run(language, "BANKLINES", config_file)
    print("Banklines done")
    run(language, "BANKEROSION", config_file)
    print("Bank erosion done")