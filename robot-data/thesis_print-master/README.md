# thesis_print - advanced print code to show more information

the real code is from https://github.com/matansar/thesis

update this code in this repository to give more information

make sure to install this before you run this tool:

```
sudo apt-get install python-pip

sudo pip install --upgrade pip

sudo pip install numpy

sudo -H pip install numpy

sudo pip install pandas

sudo pip install sklearn

sudo pip install scipy
```
in cmd.sh file we check here the manipulation scenario that gripping_without case so we put there the path's like this:

```
python software/AnomalyDetection.py -p "available runs/manipulation scenarios/gripping_without/normal/" -n "available runs/manipulation scenarios/gripping_without/test/" -c "available runs/manipulation scenarios/gripping_without/charts/" -t 3
```
let's over it one by one:

python - to run it with python

software/AnomalyDetection.py - which file to run with the python

-p "available runs/manipulation scenarios/gripping_without/normal/" - path of the good files - positives

-n "available runs/manipulation scenarios/gripping_without/test/" - path of the bad files - negatives

-c "available runs/manipulation scenarios/gripping_without/charts/" - to make graphs - charts

-t 3 - in this version this is not in used,  we take the max threshold of all the files in trainings list files that take randomaly from the positives files path

to run:

```
cd theasis_print

chmod u+x cmd.sh

./cmd.sh
```

the output will be like this:

```
kind_of_checking_file-name_of_the_file_that_checked: ['name_of_rule-name_off_the_feature(name_of_topic): number_of_line']
```
 
for example like above:
```
trainings-new_gripping_8.csv: ['coverage percentage columns-Counter(/tf): 10']
```
it will print this for the all files that checked
