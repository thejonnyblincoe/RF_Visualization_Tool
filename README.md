# RF Visualization Tool
As someone who appreciates visualization as a tool to understanding mathematical and scientific concepts, I figured making an RF Visualization Tool would come in handy.

This is a very basic GUI that allows the user to change a set of parameters to display:
-The analog signal generated (what the front end would see)
-The sample points taken of the analog signal in a time domain
-The sample points taken of the analog signal shown in the frequency domain
-The I/Q data that is abstracted from the sampled data
-The amplitude and phase of the reconstructed signal

## Bugs
-The amplitude and phase plot is not clearing when updating parameters, to work around this close the plot and reopen after setting new param.
