import jetbrains.buildServer.configs.kotlin.*

import BuildWithCommandWindow

object SignedReleaseTerminal : BuildType({
    name = "Signed release with command window"
    description = "Collect D-FAST Bank Erosion signed release with terminal window for debugging"

    artifactRules = """
        . => dfastbe-signed-%build.revisions.short%.zip
        -:dfastbe.zip
    """.trimIndent()
    buildNumberPattern = "%build.revisions.short%"

    vcs {
        root(DslContext.settingsRoot)
    }

    dependencies {
        dependency(BuildWithCommandWindow) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "dfastbe.zip!** => ."
            }
        }
        dependency(AbsoluteId("SigningAndCertificates_DFast_SigningDFastBankErosionTestCode")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "dfastbe.exe"
            }
        }
    }
})