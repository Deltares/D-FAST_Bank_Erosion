import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnText
import DFastCleanConfiguration

object BuildWithCommandWindow : BuildType({
    templates(DFastCleanConfiguration)
    id("BuildWithCommandWindow")
    name = "Build with command window"

    artifactRules = """
        dfastbe.dist => dfastbe.zip
        compilation-report.xml => .
    """.trimIndent()

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "build D-FAST BE"
            id = "build_D_FAST_BE"
            scriptContent = """
                CALL conda activate %CONDA_ENV_NAME%
                CALL .\BuildScripts\BuildDfastbe.bat
                CALL conda deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("Conda_create_environment", "Python_pip_install_poetry", "Install_dependencies_via_poetry", "Unit_test_and_code_coverage", "build_D_FAST_BE", "Conda_deactivate_and_remove_environment")
    }

    failureConditions {
        executionTimeoutMin = 90
        failOnText {
            id = "BUILD_EXT_462"
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "AssertionError"
            failureMessage = "AssertionError"
            reverse = false
            stopBuildOnFailure = true
        }
        failOnMetricChange {
            id = "BUILD_EXT_479"
            metric = BuildFailureOnMetric.MetricType.ARTIFACTS_TOTAL_SIZE
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.LESS
            compareTo = value()
            param("metricThreshold", "100MB")
        }
    }

    dependencies {
        dependency(LatexManualGeneration) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                id = "ARTIFACT_DEPENDENCY_5007"
                artifactRules = "+:*.pdf => docs/"
            }
        }
        snapshot(UnitTestsSonarCloud) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
    }
    
    disableSettings("TRIGGER_648", "Unit_test_and_code_coverage", "swabra")
})