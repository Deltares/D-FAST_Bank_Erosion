import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object PoetryTemplate : Template({
    name = "D-FAST Poetry configuration"

    artifactRules = """
        coverage.xml
        report.xml
    """.trimIndent()
    buildNumberPattern = "%build.revisions.short%"

    vcs {
        cleanCheckout = true
    }

    steps {
        script {
            name = "Install Poetry"
            id = "install_poetry"
            scriptContent = """
                set PATH=%env.PYTHON_PATH%;%env.PYTHON_PATH%\Scripts;%PATH%
                python -m pip install --upgrade pip
                python -m pip install poetry
            """.trimIndent()
        }
        script {
            name = "Create Poetry environment"
            id = "create_poetry_environment"
            scriptContent = """
                set PATH=%env.PYTHON_PATH%;%env.PYTHON_PATH%\Scripts;%PATH%
                poetry env use %env.PYTHON_PATH%\python.exe
                poetry run python --version
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                set PATH=%env.PYTHON_PATH%;%env.PYTHON_PATH%\Scripts;%PATH%
                poetry install
                poetry show
            """.trimIndent()
        }
        script {
            name = "Cleanup Poetry environment"
            id = "cleanup_poetry_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                set PATH=%env.PYTHON_PATH%;%env.PYTHON_PATH%\Scripts;%PATH%
                poetry env remove --all
            """.trimIndent()
        }
    }

    requirements {
        exists("env.PYTHON_PATH")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
