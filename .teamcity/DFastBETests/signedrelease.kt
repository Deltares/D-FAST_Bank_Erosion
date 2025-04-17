import jetbrains.buildServer.configs.kotlin.*

object SignedRelease : BuildType({
    id("SignedRelease")
    name = "Signed release with command window"

    artifactRules = """
        . => dfastbe-signed-${DFast_DFastBankErosion_BuildWithCommandWindow.depParamRefs["build.revisions.short"]}.zip
        -:dfastbe.zip
    """.trimIndent()
    buildNumberPattern = "${DFast_DFastBankErosion_BuildWithCommandWindow.depParamRefs["build.revisions.short"]}"

    vcs {
        root(DslContext.settingsRoot)
    }

    dependencies {
        dependency(DFast_DFastBankErosion_BuildWithCommandWindow) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "dfastbe.zip!** => ."
            }
        }
        dependency(AbsoluteId("SigningAndCertificates_DFast_SigningDFastBankErosion")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "dfastbe.exe"
            }
        }
    }
})