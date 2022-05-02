## How to run this code?
Go in DeepOrchestration/source_separation folder.
Launch using python script

Where 'script' can be:
- create_targets
- separate_targets
- orchestrate_targets
- evaluate (separation or orchestration)

## Dataset folder organization
Targets metadata.json files contain the offsets used to create the targets and needed to evaluate separation.

	/dataset_folder
		orchestrated/
			2sources/
				...
			3sources/
				...
			4sources/
				full_target_orch.wav
				/ground_truth/
					source1_orch.wav
					source2_orch.wav
					source3_orch.wav
					source4_orch.wav
				/method1/
					source1_orch.wav
					source2_orch.wav
					...
				/method2/
					...
		samples/
		separated/
			2sources/
				method1/
					target1/
						source1.wav
						source2.wav
				method2/
			3sources/
				method1/
					target1/
						source1.wav
			4sources/
				...
		targets/
			2sources/
				metadata.json
				target1.wav
				...
			3sources/
				...
			4sources/
				...

## TODO list

 - [ ] Simplify targets names / resolve problem with "*" character in filenames.
 - [ ] Make script to plot separation results as X axis and orchestration results as Y axis to show correlation between the two.