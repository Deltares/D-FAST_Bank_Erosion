import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object PoetryTemplate : Template({
    name = "D-FAST Poetry configuration"

    buildNumberPattern = "%build.revisions.short%"

    vcs {
        cleanCheckout = true
    }

    steps {
        script {
            name = "Install Poetry"
            id = "install_poetry"
            scriptContent = """
                curl -sSL https://install.python-poetry.org | python -
                set PATH=%APPDATA%\Python\Scripts;%PATH%
            """.trimIndent()
        }
        script {
            name = "Create Poetry environment"
            id = "create_poetry_environment"
            scriptContent = """
                set PATH=%APPDATA%\Python\Scripts;%PATH%
                poetry env use python
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                set PATH=%APPDATA%\Python\Scripts;%PATH%
                poetry install
                poetry show
            """.trimIndent()
        }
        script {
            name = "Cleanup Poetry environment"
            id = "cleanup_poetry_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                set PATH=%APPDATA%\Python\Scripts;%PATH%
                poetry env remove --all
            """.trimIndent()
        }
    }

    requirements {
        exists("env.python3913")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
