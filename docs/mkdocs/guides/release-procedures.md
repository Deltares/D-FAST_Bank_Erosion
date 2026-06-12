# Release procedures

This guide describes the manual steps for assembling a D-FAST Bank Erosion release once the
TeamCity builds have completed.

!!! note
    The TeamCity links below are pinned to the `release/3.1.0` branch through the `branch=` query
    parameter. Replace it with the branch of the release you are assembling.

## Steps

1. Download the **signed release executables** from the artifacts of the TeamCity *Signed Release*
   build ([`DFast_DFastBankErosion_SignedRelease`][signed-release]).
2. Download the **documentation** — the release notes, the technical reference manual and the user
   manual — from the artifacts of the TeamCity *LaTeX Manual* build
   ([`DFast_DFastBankErosion_LatexManual`][latex-manual]).
3. Create a **new release folder**, named after the release version, on the shared drive:
   `P:\1209447-kpp-hydraulicaprogrammatuur\D-FAST\<release>`.

[signed-release]: https://dpcbuild.deltares.nl/buildConfiguration/DFast_DFastBankErosion_SignedRelease?branch=release%2F3.1.0&buildTypeTab=overview
[latex-manual]: https://dpcbuild.deltares.nl/buildConfiguration/DFast_DFastBankErosion_LatexManual?branch=release%2F3.1.0&buildTypeTab=overview
