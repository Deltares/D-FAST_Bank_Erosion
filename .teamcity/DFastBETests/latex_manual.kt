import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnText
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object LatexManual : BuildType({
    name = "Latex Manual Generation"
    description = "Generate the D-FAST Bank Erosion user manual and technical reference using LaTeX."

    artifactRules = """
        +:%artifact_path%*.pdf
        +:%artifact_path%*.log
    """.trimIndent()
    buildNumberPattern = "%build.revisions.short%"

    params {
        param("artifact_path", "docs/end-user-docs/")
        param("file_name_of_techref", "dfastbe_techref")
        param("file_name_of_usermanual", "dfastbe_usermanual")
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Write gitsettings file"
            id = "write_gitsettings_file"
            workingDir = "docs/end-user-docs"
            scriptContent = """
                echo \def\@gitrepository{Unknown repository} > gitsettings
                echo \def\@gitbranch{Unknown branch} >> gitsettings
                echo \def\@githashlong{%build.revisions.revision%} >> gitsettings
                echo \def\@githashshort{%build.revisions.short%} >> gitsettings
            """.trimIndent()
        }
        script {
            name = "Generate User Manual"
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            workingDir = "docs/end-user-docs"
            scriptContent = """
                pdflatex %file_name_of_usermanual%
                bibtex %file_name_of_usermanual%
                pdflatex %file_name_of_usermanual%
                pdflatex %file_name_of_usermanual%
            """.trimIndent()
        }
        script {
            name = "Generate Technical reference"
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            workingDir = "docs/end-user-docs"
            scriptContent = """
                pdflatex %file_name_of_techref%
                bibtex %file_name_of_techref%
                pdflatex %file_name_of_techref%
                pdflatex %file_name_of_techref%
            """.trimIndent()
        }
    }

    failureConditions {
        failOnText {
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "Output written on %file_name_of_techref%.pdf"
            failureMessage = "Technical reference generation failed"
            reverse = true
        }
        failOnText {
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "Output written on %file_name_of_usermanual%.pdf"
            failureMessage = "User manual generation failed"
            reverse = true
        }
    }

    features {
        swabra {
            forceCleanCheckout = true
        }
        commitStatusPublisher {
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "%github_deltares-research_private_ssh_key_password%"
                }
            }
        }
    }

    requirements {
        exists("WindowsSDKv10.0")
    }
})