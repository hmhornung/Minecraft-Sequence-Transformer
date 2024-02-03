Transformer model combined with learned 3-Dimensional Positional Encodings for the purpose of doing next-block sequence prediction in Minecraft.

The first iteration will be a Transformer model with an input sequence of 5x5x5 array of minecraft blocks that predicts a 1x5x5 array of minecraft blocks adjacent to the end on the positive x (1st) axis of the input array.
These 5x5x5 input and 1x5x5 output arrays will be in a dataset made from taking pieces of the specified dimension (6x5x5) from minecraft schematics.
