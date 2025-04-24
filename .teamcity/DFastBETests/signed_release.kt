import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object SignedRelease : BuildType({
    name = "Signed release"
    description = "Collect D-FAST Bank Erosion signed release"

    artifactRules = """
        . => dfastbe-signed-%build.revisions.short%.zip
        -:dfastbe.zip
    """.trimIndent()
    buildNumberPattern = "%build.revisions.short%"

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Move CLI version of dfastbe"
            id = "Move_CLI_version_of_dfastbe"
            scriptContent = """
                move dfastbe_cli\dfastbe.exe dfastbe_cli.exe
                rmdir dfastbe_cli
            """.trimIndent()
        }
    }

    dependencies {
        snapshot(BuildTerminal) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(AbsoluteId("SigningAndCertificates_DFast_SigningDFastBankErosionTestCode")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "dfastbe.exe => dfastbe_cli"
            }
        }
        dependency(AbsoluteId("SigningAndCertificates_DFast_SigningDFastBankErosionSuppressCommandWindowTestCode")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "dfastbe.exe"
            }
        }
        artifacts(BuildMain) {
            artifactRules = "dfastbe.zip!** => ."
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})