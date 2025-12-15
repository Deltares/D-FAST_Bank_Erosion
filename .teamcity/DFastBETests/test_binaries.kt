import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import PoetryTemplate

object TestBinaries : BuildType({
    templates(PoetryTemplate)
    name = "Distribution Tests"
    description = "Test D-FAST Bank Erosion binaries."

    buildNumberPattern = "%build.revisions.short%"

    vcs {
        root(DslContext.settingsRoot)
    }

    params {
        param("test_dir", "tests/test_binaries/")
    }

    steps {
        script {
            name = "Get folder listing"
            id = "Get_folder_listing"
            scriptContent = """
                echo "Listing current folder"
                dir
                echo "------------------"
            """.trimIndent()
        }
        script {
            name = "Validate distribution"
            id = "Validate_distribution"
            scriptContent = """
                set PATH=%APPDATA%\Python\Scripts;%PATH%
                for /f "delims=" %%i in ('poetry env info --path') do set POETRY_ENV_PATH=%%i
                CALL "%POETRY_ENV_PATH%\Scripts\activate.bat"
                pytest -v %test_dir% --no-cov
                deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("install_poetry", "create_poetry_environment", "Install_dependencies_via_poetry", "Get_folder_listing", "Validate_distribution", "cleanup_poetry_environment")
    }

    failureConditions {
        failOnMetricChange {
            metric = BuildFailureOnMetric.MetricType.TEST_COUNT
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.LESS
            compareTo = value()
        }
    }


    if (DslContext.getParameter("environment") == "production") {
        triggers {
            vcs {
                branchFilter = "+:*"
            }
        }
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
    }

    dependencies {
        dependency(SignedReleaseTerminal) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "*.zip!** => dfastbe.dist"
            }
        }
    }
})