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
                mkdir C:\poetry-temp
                curl -sSL https://install.python-poetry.org -o C:\poetry-temp\install-poetry.py
                set POETRY_HOME=C:\poetry-temp\poetry
                %env.PYTHON_PATH%\python.exe C:\poetry-temp\install-poetry.py --force
                del C:\poetry-temp\install-poetry.py
            """.trimIndent()
        }
        script {
            name = "Create Poetry environment"
            id = "create_poetry_environment"
            scriptContent = """
                chcp 65001
                C:\poetry-temp\poetry\bin\poetry.exe env use %env.PYTHON_PATH%\python.exe
                C:\poetry-temp\poetry\bin\poetry.exe run python --version
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                chcp 65001
                C:\poetry-temp\poetry\bin\poetry.exe install
                C:\poetry-temp\poetry\bin\poetry.exe show
            """.trimIndent()
        }
        script {
            name = "Cleanup Poetry environment and installation"
            id = "cleanup_poetry_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                C:\poetry-temp\poetry\bin\poetry.exe env remove --all
                rmdir /S /Q C:\poetry-temp
            """.trimIndent()
        }
    }

    requirements {
        exists("env.PYTHON_PATH")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
