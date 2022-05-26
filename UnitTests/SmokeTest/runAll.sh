#!/bin/bash

{ 
    # Try (linux)
    source /home/$USER/anaconda3/bin/activate bayes_opt
} || {
    # Except (mac)
    source /Users/$USER/opt/anaconda3/bin/activate bayes_opt
}

# Actually run the smoke tests
for file in *.py
do 
    python $file
done

rm Example_file.p
