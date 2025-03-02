My attempts at writing an einstein field equation solver.

DISCLAIMER: This project is under active work and currently doesn't work.

TODO:
- Investigate divergence:
    - Look at spectral derivative code in case of improper dealiasing and/or normalisation.
- Refactoring:
    - Move away from 1 + log() slicing for gauge condition.
    - Investigate using BSSN and Z4c formulations.
- Parrelelize more hot loops.
