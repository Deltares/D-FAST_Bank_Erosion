
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.*
import UnitTestsSonarCloud
version = "2024.12"
project {
    description = "D-FAST Bank Erosion"

    params {
        param("CONDA_ENV_NAME", "python_3_11_12-dfastbe")
        param("python.version", "3.11.12")
        param("SonarSource", "dfastbe")
        param("SonarProjectKey", "Deltares_D-FAST_Bank_Erosion")
        password("sonar_server", "credentialsJSON:b6bc3a37-8077-45db-9f3c-da2b5db2e8ca")
    }

    buildType(UnitTestsSonarCloud)
}