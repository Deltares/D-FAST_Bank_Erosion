import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import CondaTemplate

object TestBinaries : BuildType({
    templates(CondaTemplate)
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
                rem echo on
                rem CALL poetry env use %%python3913%%\python.exe
                CALL conda activate %CONDA_ENV_NAME%
                CALL poetry run pytest -v %test_dir% --no-cov
                CALL conda deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("Conda_create_environment", "Python_pip_install_poetry", "Install_dependencies_via_poetry", "Get_folder_listing", "Validate_distribution", "Conda_deactivate_and_remove_environment")
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