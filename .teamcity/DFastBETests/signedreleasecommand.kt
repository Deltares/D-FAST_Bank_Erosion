import jetbrains.buildServer.configs.kotlin.*

import BuildWithCommandWindow

object SignedReleaseCommand : BuildType({
    id("SignedReleaseCommand")
    name = "Signed release with command window"

    artifactRules = """
        . => dfastbe-signed-${BuildWithCommandWindow.depParamRefs["build.revisions.short"]}.zip
        -:dfastbe.zip
    """.trimIndent()
    buildNumberPattern = "${BuildWithCommandWindow.depParamRefs["build.revisions.short"]}"

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