
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.*
import UnitTestsSonarCloud
import DFastCleanConfiguration
import LatexManualGeneration
import SignedReleaseCommand
import BuildWithCommandWindow
import BuildWithoutCommandWindow
import DistributionTests
import SignedRelease

version = "2025.03"

project {
    description = "D-FAST Bank Erosion"

    params {
        param("CONDA_ENV_NAME", "python_3_11_12-dfastbe")
        param("python.version", "3.11.12")
        param("SonarSource", "dfastbe")
        param("SonarProjectKey", "Deltares_D-FAST_Bank_Erosion")
        password("sonar_server", "credentialsJSON:b6bc3a37-8077-45db-9f3c-da2b5db2e8ca")
    }

    template(DFastCleanConfiguration)

    buildType(UnitTestsSonarCloud)
    buildType(LatexManualGeneration)
    buildType(SignedReleaseCommand)
    buildType(BuildWithCommandWindow)
    buildType(DistributionTests)
    buildType(SignedRelease)
    buildType(BuildWithoutCommandWindow)

    buildTypesOrder = arrayListOf(
        LatexManualGeneration,
        UnitTestsSonarCloud,
        BuildWithCommandWindow,
        SignedReleaseCommand,
        DistributionTests,
        BuildWithoutCommandWindow,
        SignedRelease
    )
}