import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnText

object BuildMain : BuildType({
    templates(CondaTemplate)
    name = "Build without command window"

    artifactRules = """
        dfastbe.dist => dfastbe.zip
        compilation-report.xml =>.
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
                CALL .\BuildScripts\BuildDfastbe_no_command_window.bat
                CALL conda deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("Conda_create_environment", "Python_pip_install_poetry", "Install_dependencies_via_poetry", "build_D_FAST_BE", "Conda_deactivate_and_remove_environment")
    }

    failureConditions {
        failOnText {
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "AssertionError"
            failureMessage = "AssertionError"
            reverse = false
            stopBuildOnFailure = true
        }
    }

    dependencies {
        snapshot(TestBinaries) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(LatexManualGeneration) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:*.pdf => docs/"
            }
        }
    }
})