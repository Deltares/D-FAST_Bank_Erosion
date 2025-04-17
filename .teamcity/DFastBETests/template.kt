import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object DFastCleanConfiguration : Template({
    name = "D-FAST Clean configuration"

    artifactRules = """
        coverage.xml
        report.xml
    """.trimIndent()
    buildNumberPattern = "%build.revisions.short%"

    params {
        param("SonarProjectKey", "Deltares_D-FAST_Morphological_Impact")
        param("SonarSource", "dfastmi")
    }

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
            scriptContent = "CALL conda create -v -y -n %CONDA_ENV_NAME% python=3.9.13"
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
                CALL python -m poetry install
                CALL python -m poetry show
                CALL conda deactivate
            """.trimIndent()
        }
        script {
            name = "Unit test and code coverage"
            id = "Unit_test_and_code_coverage"
            scriptContent = """
                CALL conda activate %CONDA_ENV_NAME%
                CALL poetry run pytest --junitxml="report.xml" --cov=%COVERAGE_LOC% --cov-report=xml tests/
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

    triggers {
        vcs {
            id = "TRIGGER_648"
        }
    }

    features {
        swabra {
            id = "swabra"
            forceCleanCheckout = true
        }
    }

    requirements {
        exists("env.python3913", "RQ_368")
        contains("teamcity.agent.jvm.os.name", "Windows", "RQ_372")
    }
})
