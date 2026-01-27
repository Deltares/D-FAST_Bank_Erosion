
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.*
import UnitTests
import PoetryTemplate
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
        param("python.version", "3.11.12")
        param("poetry.path", "%teamcity.agent.home.dir%\\..\\poetry")
        param("SonarSource", "dfastbe")
        param("SonarProjectKey", "Deltares_D-FAST_Bank_Erosion")
    }

    template(PoetryTemplate)

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