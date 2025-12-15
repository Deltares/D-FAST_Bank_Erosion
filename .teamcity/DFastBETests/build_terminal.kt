import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnText
import PoetryTemplate

object BuildTerminal : BuildType({
    templates(PoetryTemplate)
    name = "Build with command window"
    description = "Build D-FAST Bank Erosion with terminal window for debugging"

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
                %APPDATA%\Python\Scripts\poetry.exe run cmd /c .\BuildScripts\BuildDfastbe.bat
                if errorlevel 1 exit /b 1
            """.trimIndent()
        }
        stepsOrder = arrayListOf("install_poetry", "create_poetry_environment", "Install_dependencies_via_poetry", "build_D_FAST_BE", "cleanup_poetry_environment")
    }

    failureConditions {
        executionTimeoutMin = 90
        failOnText {
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "AssertionError"
            failureMessage = "AssertionError"
            reverse = false
            stopBuildOnFailure = true
        }
        failOnMetricChange {
            metric = BuildFailureOnMetric.MetricType.ARTIFACTS_TOTAL_SIZE
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.LESS
            compareTo = value()
            param("metricThreshold", "100MB")
        }
    }

    dependencies {
        dependency(LatexManual) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:*.pdf => docs/"
            }
        }
        snapshot(UnitTests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
    }
})