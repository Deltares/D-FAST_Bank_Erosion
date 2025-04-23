import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object CondaTemplate : Template({
    name = "D-FAST Clean configuration"

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
            name = "Conda remove environment"
            id = "Conda_remove_environment"
            enabled = false
            scriptContent = "CALL conda remove --name %CONDA_ENV_NAME% --all --force-remove"
        }
        script {
            name = "Conda create environment"
            id = "Conda_create_environment"
            scriptContent = """
                rmdir /S /Q %CONDA_PATH%\\%CONDA_ENV_NAME%
                CALL conda create -v -y -n %CONDA_ENV_NAME% python=%python.version%
            """.trimIndent()
        }
        script {
            name = "Python pip install poetry"
            id = "Python_pip_install_poetry"
            scriptContent = """
                CALL conda activate %CONDA_ENV_NAME%
                rem pip install --upgrade pip
                rem pip install poetry --user
                CALL python -m pip install --upgrade pip
                CALL python -m pip install poetry --user
                CALL conda deactivate
            """.trimIndent()
        }
        script {
            name = "Install dependencies via poetry"
            id = "Install_dependencies_via_poetry"
            scriptContent = """
                CALL conda activate %CONDA_ENV_NAME%
                CALL pip install --upgrade virtualenv
                CALL python -m poetry install
                CALL python -m poetry show
                CALL conda deactivate
            """.trimIndent()
        }
        script {
            name = "Conda deactivate and remove environment"
            id = "Conda_deactivate_and_remove_environment"
            executionMode = BuildStep.ExecutionMode.ALWAYS
            scriptContent = """
                CALL conda deactivate
                rem CALL conda env remove -y --name %CONDA_ENV_NAME%
                CALL conda remove --name %CONDA_ENV_NAME% --all --force-remove
            """.trimIndent()
        }
    }

    requirements {
        exists("env.python3913")
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})
