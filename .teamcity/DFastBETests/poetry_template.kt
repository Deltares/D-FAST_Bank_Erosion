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
        param("poetry.temp.dir", "%teamcity.build.tempDir%\\poetry-temp")
        param("poetry.exe.path", "%poetry.temp.dir%\\poetry\\bin\\poetry.exe")
    }

    vcs {
        cleanCheckout = true
    }

    steps {
        script {
            name = "Install Poetry standalone"
            id = "install_poetry"
            scriptContent = """
                mkdir %poetry.temp.dir%
                curl -sSL https://install.python-poetry.org -o %poetry.temp.dir%\install-poetry.py
                set POETRY_HOME=%poetry.temp.dir%\poetry
                %env.PYTHON_PATH%\python.exe %poetry.temp.dir%\install-poetry.py --force
                del %poetry.temp.dir%\install-poetry.py
            """.trimIndent()
        }
        script {
            name = "Create Poetry environment"
            id = "create_poetry_environment"
            scriptContent = """
                %poetry.exe.path% env use %env.PYTHON_PATH%\python.exe
                %poetry.exe.path% run python --version
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                %poetry.exe.path% install
            """.trimIndent()
        }
        script {
            name = "Cleanup Poetry environment and installation"
            id = "cleanup_poetry_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                %poetry.exe.path% env remove --all
                rmdir /S /Q %poetry.temp.dir%
            """.trimIndent()
        }
    }

    requirements {
        exists("env.PYTHON_PATH")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
