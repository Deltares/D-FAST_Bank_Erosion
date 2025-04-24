
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.*
import UnitTests
import CondaTemplate
import LatexManual
import SignedReleaseTerminal
import BuildTerminal
import BuildMain
import TestBinaries
import SignedRelease

version = "2025.03"

project {
    description = "D-FAST Bank Erosion"

    params {
        param("CONDA_ENV_NAME", "python-dfastbe")
        param("CONDA_PATH", "D:\\ProgramData\\Miniforge3\\envs")
        param("python.version", "3.11.12")
        param("SonarSource", "dfastbe")
        param("SonarProjectKey", "Deltares_D-FAST_Bank_Erosion")
    }

    template(CondaTemplate)

    buildType(UnitTests)
    buildType(LatexManual)
    buildType(SignedReleaseTerminal)
    buildType(BuildTerminal)
    buildType(TestBinaries)
    buildType(SignedRelease)
    buildType(BuildMain)

    buildTypesOrder = arrayListOf(
        LatexManual,
        UnitTests,
        BuildTerminal,
        SignedReleaseTerminal,
        TestBinaries,
        BuildMain,
        SignedRelease
    )
}