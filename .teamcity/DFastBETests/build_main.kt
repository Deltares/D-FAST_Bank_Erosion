import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnText

object BuildMain : BuildType({
    templates(PoetryTemplate)
    name = "Build without command window"
    description = "Build D-FAST Bank Erosion without terminal window as main distribution"

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
                set PATH=%APPDATA%\Python\Scripts;%PATH%
                for /f "delims=" %%i in ('poetry env info --path') do set POETRY_ENV_PATH=%%i
                CALL "%POETRY_ENV_PATH%\Scripts\activate.bat"
                CALL .\BuildScripts\BuildDfastbe_no_command_window.bat
                deactivate
            """.trimIndent()
        }
        stepsOrder = arrayListOf("install_poetry", "create_poetry_environment", "Install_dependencies_via_poetry", "build_D_FAST_BE", "cleanup_poetry_environment")
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
        dependency(LatexManual) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:*.pdf => docs/"
            }
        }
    }
})