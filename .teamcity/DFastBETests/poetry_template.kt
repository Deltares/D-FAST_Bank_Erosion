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
            name = "Install Poetry standalone"
            id = "install_poetry"
            scriptContent = """
                curl -sSL https://install.python-poetry.org -o install-poetry.py
                %env.PYTHON_PATH%\python.exe install-poetry.py
                del install-poetry.py
            """.trimIndent()
        }
        script {
            name = "Create Poetry environment"
            id = "create_poetry_environment"
            scriptContent = """
                %APPDATA%\Python\Scripts\poetry.exe env use %env.PYTHON_PATH%\python.exe
                %APPDATA%\Python\Scripts\poetry.exe run python --version
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                %APPDATA%\Python\Scripts\poetry.exe install
                %APPDATA%\Python\Scripts\poetry.exe show
            """.trimIndent()
        }
        script {
            name = "Cleanup Poetry environment"
            id = "cleanup_poetry_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                %APPDATA%\Python\Scripts\poetry.exe env remove --all
            """.trimIndent()
        }
    }

    requirements {
        exists("env.PYTHON_PATH")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
