import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import PoetryTemplate

object UnitTests : BuildType({
    templates(PoetryTemplate)
    name = "Unit Tests + SonarCloud"
    description = "Run unit tests and SonarCloud analysis."

    artifactRules = """
        coverage.xml
        report.xml
        packages.txt
    """.trimIndent()

    params {
        param("COVERAGE_LOC", "src/dfastbe")
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Unit test and code coverage"
            id = "Unit_test_and_code_coverage"
            scriptContent = """
                set PATH=%APPDATA%\Python\Scripts;%PATH%
                for /f "delims=" %%i in ('poetry env info --path') do set POETRY_ENV_PATH=%%i
                CALL "%POETRY_ENV_PATH%\Scripts\activate.bat"
                pytest --junitxml="report.xml" --cov=%COVERAGE_LOC% --cov-report=xml tests/ -m "not binaries"
                deactivate
            """.trimIndent()
        }
        step {
            name = "SonarCloud analysis"
            id = "SonarCloud_analysis"
            type = "sonar-plugin"
            enabled = false
            param("sonarProjectName", "D-FAST_Bank_Erosion")
            param("additionalParameters", """
                "-X"
                "-Dsonar.branch.name=%teamcity.build.branch%"
                "-Dsonar.python.coverage.reportPaths=coverage.xml"
                "-Dsonar.python.xunit.reportPath=report.xml"
                "-Dsonar.organization=deltares"
                "-Dsonar.token=%sonar_token%"
            """.trimIndent())
            param("sonarProjectKey", "%SonarProjectKey%")
            param("sonarServer", "41dee3f5-7fe2-478a-865e-d0b26dba20f1")
        }
        stepsOrder = arrayListOf("install_poetry", "create_poetry_environment", "Install_dependencies_via_poetry", "Unit_test_and_code_coverage", "SonarCloud_analysis", "cleanup_poetry_environment")
    }

    features {
        swabra {
            forceCleanCheckout = true
        }
        commitStatusPublisher {
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "%github_deltares-service-account_access_token%"
                }
            }
        }
        perfmon {
            id = "perfmon"
        }
        feature {
            id = "JetBrains.SonarQube.BranchesAndPullRequests.Support"
            type = "JetBrains.SonarQube.BranchesAndPullRequests.Support"
            param("provider", "GitHub")
        }
    }
})