import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import CondaTemplate

object UnitTestsSonarCloud : BuildType({
    templates(CondaTemplate)
    id("UnitTestsSonarCloud")
    name = "Unit Tests + SonarCloud"

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
                CALL conda activate %CONDA_ENV_NAME%
                CALL poetry run pytest --junitxml="report.xml" --cov=%COVERAGE_LOC% --cov-report=xml tests/ -m "not binaries"
                CALL conda deactivate
            """.trimIndent()
        }
        step {
            name = "SonarCloud analysis"
            id = "SonarCloud_analysis"
            type = "sonar-plugin"
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
        stepsOrder = arrayListOf("Conda_create_environment", "Python_pip_install_poetry", "Install_dependencies_via_poetry", "Unit_test_and_code_coverage", "SonarCloud_analysis", "Conda_deactivate_and_remove_environment")
    }

    features {
        swabra {
            forceCleanCheckout = true
        }
        commitStatusPublisher {
            id = "BUILD_EXT_316"
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "%github_teamcity_commit_status_token%"
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