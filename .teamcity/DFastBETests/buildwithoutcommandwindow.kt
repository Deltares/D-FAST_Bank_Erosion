import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnText

object BuildWithoutCommandWindow : BuildType({
    templates(DFastCleanConfiguration)
    id("BuildWithoutCommandWindow")
    name = "Build without command window"

    artifactRules = """
        dfastbe.dist => dfastbe.zip
        compilation-report.xml =>.
    """.trimIndent()

    params {
        param("CONDA_ENV_NAME", "python_3_9_13-dfastbe")
        param("SonarProjectKey", "")
        param("SonarSource", "")
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "build D-FAST BE"
            id = "build_D_FAST_BE"
            scriptContent = """
                CALL conda activate %CONDA_ENV_NAME%
                CALL .\BuildScripts\BuildDfastbe_no_command_window.bat
                CALL conda deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("Conda_create_environment", "Python_pip_install_poetry", "Install_dependencies_via_poetry", "Unit_test_and_code_coverage", "build_D_FAST_BE", "Conda_deactivate_and_remove_environment")
    }

    failureConditions {
        failOnText {
            id = "BUILD_EXT_478"
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "AssertionError"
            failureMessage = "AssertionError"
            reverse = false
            stopBuildOnFailure = true
        }
    }

    dependencies {
        snapshot(DistributionTests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(LatexManualGeneration) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                id = "ARTIFACT_DEPENDENCY_12900"
                artifactRules = "+:*.pdf => docs/"
            }
        }
    }
    
    disableSettings("TRIGGER_648", "Unit_test_and_code_coverage", "swabra")
})