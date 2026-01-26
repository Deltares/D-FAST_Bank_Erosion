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

    params {
        param("POETRY_HOME", "%teamcity.build.tempDir%\\poetry-temp")
        param("POETRY_EXE", "%POETRY_HOME%\\poetry\\bin\\poetry.exe")
    }

    vcs {
        cleanCheckout = true
    }

    steps {
        script {
            name = "Install Poetry standalone"
            id = "install_poetry"
            scriptContent = """
                mkdir %POETRY_HOME%
                curl -sSL https://install.python-poetry.org -o %POETRY_HOME%\install-poetry.py
                set POETRY_HOME=%POETRY_HOME%\poetry
                %env.PYTHON_PATH%\python.exe %POETRY_HOME%\install-poetry.py --force
                del %POETRY_HOME%\install-poetry.py
            """.trimIndent()
        }
        script {
            name = "Create Poetry environment"
            id = "create_poetry_environment"
            scriptContent = """
                %POETRY_EXE% env use %env.PYTHON_PATH%\python.exe
                %POETRY_EXE% run python --version
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                %POETRY_EXE% install
            """.trimIndent()
        }
        script {
            name = "Cleanup Poetry environment and installation"
            id = "cleanup_poetry_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                %POETRY_EXE% env remove --all
                rmdir /S /Q %POETRY_HOME%
            """.trimIndent()
        }
    }

    requirements {
        exists("env.PYTHON_PATH")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
