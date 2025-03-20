from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from dfastbe.cmd import run


def test_bank_lines():
    language = "UK"
    run_mode = "BANKLINES"
    config_file = "tests/data/integration_test2/Meuse_manual.cfg"
    run(language, run_mode, config_file)
    file_1 = Path(
        r"tests\\data\\integration_test2\\output\\banklines\\raw_detected_bankline_fragments.shp"
    )
    file_2 = Path(r"tests\\data\\integration_test2\\output\\banklines\\bank_areas.shp")
    file_3 = Path(
        r"tests\\data\\integration_test2\\output\\banklines\\bankline_fragments_per_bank_area.shp"
    )
    file_4 = Path(r"tests\\data\\integration_test2\\output\\banklines\\bankfile.shp")
    assert file_1.exists()
    assert file_2.exists()
    assert file_3.exists()
    assert file_4.exists()
    fig_1 = Path(
        r"tests\\data\\integration_test2\\output\\figures\\1_banklinedetection.png"
    )
    assert fig_1.exists()
