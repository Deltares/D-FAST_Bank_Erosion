import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import CondaTemplate

object DistributionTests : BuildType({
    templates(CondaTemplate)
    id("DistributionTests")
    name = "Distribution Tests"

    buildNumberPattern = "${BuildWithCommandWindow.depParamRefs["build.revisions.short"]}"

    vcs {
        root(DslContext.settingsRoot)
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
                CALL poetry run pytest -v tests/test_binaries/ --no-cov
                CALL conda deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("Conda_create_environment", "Python_pip_install_poetry", "Install_dependencies_via_poetry", "Get_folder_listing", "Validate_distribution", "Conda_deactivate_and_remove_environment")
    }

    failureConditions {
        failOnMetricChange {
            id = "BUILD_EXT_469"
            metric = BuildFailureOnMetric.MetricType.TEST_COUNT
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.LESS
            compareTo = value()
        }
    }


    triggers {
        vcs {
            id = "TRIGGER_648"
            enabled = false
        }
    }


    features {
        swabra {
            forceCleanCheckout = true
        }
        commitStatusPublisher {
            id = "BUILD_EXT_315"
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "%github_teamcity_commit_status_token%"
                }
            }
        }
    }

    dependencies {
        dependency(SignedReleaseCommand) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                id = "ARTIFACT_DEPENDENCY_5392"
                artifactRules = "*.zip!** => dfastbe.dist"
            }
        }
    }
})