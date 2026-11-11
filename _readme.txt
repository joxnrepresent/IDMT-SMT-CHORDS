This dataset contains chords wav files resampled from midi files using various sampler instruments: 

________________________________________________________________________________________________________________

Wav File Naming convention: 

<DAW used>_<dir of VST/sampler>_<name of VST/sampler>

Example: 

logic_pro_x_piano_Steinway_Grand_Piano
________________________________________________________________________________________________________________

Annotation naming convention:

root : shorthand (extra-intervals) / Bass  ----> For Example for C/E  ---->  C:maj7/3


|------------------------------------------------------------------------------------
|Chord Type | short-hand| Example     	| Bass/inversions Examples
|------------------------------------------------------------------------------------
|major      | maj  	| C:maj		| c:maj/3   (C/E,     C 1st inversion)
|minor      | min	| C:min		| C:min/5   (Cm/G,    C 2st inversion)
|major7     | maj7	| C:maj7	| C:maj7/7  (Cmaj7/B, C 3st inversion)
|minor7     | min7	| C:min7	| C:min7/7  (Cmin7/B, C 3st inversion)
|powerchord | 5 	| C:5 		| C:5/5     (C5/G,    C 1st inversion)
|dominant 7 | 7		| C:7		| C:7/5     (C7/G,    C 2st inversion)
|m7b5       | hdim7	| C:hdim7	| C:hdim7/5 (Cm7b5/G, C 2st inversion)
|------------------------------------------------------------------------------------

________________________________________________________________________________________________________________

For any questions, please contact datasets@idmt.fraunhofer.de

Reference

Christon-Ragavan Nadar, Jakob Abeﬂer, Sascha Grollmisch: Towards CNN-based acoustic modeling of seventh chords for automatic chord recognition, Proceedings of the 16th International Conference on Sound And Music Computing (SMC), M·laga, Spain, 2019.
