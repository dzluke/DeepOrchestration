# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!


### The online editing link is here:
https://www.overleaf.com/4125771964jsvptcvnzdwj

### Generate the dataset
- download the raw data from http://www.orch-idea.org/ (I choose StaticSOL but you can choose other datasets)
- select all ord files played by ['Vc', 'Fl', 'Va', 'Vn', 'Ob', 'BTb',
       'Cb', 'ClBb', 'Hn', 'TpC', 'Bn', 'Tbn'] and put these files in a shared directory
- call method `show_all_class_num` in `process_OrchDB.py` to get the `class.index`
- set paras in `process_OrchDB.py` 
- call method `random_combine` to generate new datasets

### Train a network
Just run `python main.py`. Before training, make sure that the class number and the dictionary `db` are right. If it is a new dataset, call method `show_all_class_num` in `process_OrchDB.py` to get the new class number and call method `stat_test_db` in `process_OrchDB.py` to get the new `db`.
 