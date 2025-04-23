
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.*
import UnitTestsSonarCloud
import CondaTemplate
import LatexManualGeneration
import SignedReleaseCommand
import BuildWithCommandWindow
import BuildWithoutCommandWindow
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

    buildType(UnitTestsSonarCloud)
    buildType(LatexManualGeneration)
    buildType(SignedReleaseCommand)
    buildType(BuildWithCommandWindow)
    buildType(TestBinaries)
    buildType(SignedRelease)
    buildType(BuildWithoutCommandWindow)

    buildTypesOrder = arrayListOf(
        LatexManualGeneration,
        UnitTestsSonarCloud,
        BuildWithCommandWindow,
        SignedReleaseCommand,
        TestBinaries,
        BuildWithoutCommandWindow,
        SignedRelease
    )
}