## Folder organization

Subtargets are samples used to create targets, that are mixture of 2-4 subtargets.
Targets metadata.json files contain the offsets used to create the targets and needed
to evaluate separation.

	/dataset_folder
		separated/
			2_sources/
				demucs/
				nmf/
				google/
				open-unmix/
			3_sources/
				...
			4_sources/
				...
		subtargets/
		targets/
			2_sources/
				metadata.json
				target1.wav
				...
			3_sources/
				metadata.json
			4_sources/

## TODO list

 - [ ] Simplify targets names / resolve problem with "*" character in filenames.
 - [ ] Make script to plot separation results as X axis and orchestration results as Y axis to show correlation between the two.