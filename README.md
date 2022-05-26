# Bayes_Opt

Much of my work is still in private repositories not readily visible. 
However, my boss has kindly allowed me to post some of the code I've written
so I can get some credit before the papers are published. This is one of 
those repositories.

# Bayesian Optimization

This is an implementation of Bayesian Optimization, which functions by 
approximating our true response curve with a
Gaussian process (random function). This is referred to as a surrogate
model. As more and more data points are sampled we achieve a better
approximation to the response curve.

The other component of Bayesian optimization is the acqusition function,
which decides where we should next sample from. This seeks to balance
exploration (maybe radically new parameters yield better results) with
exploitation (fine-tuning parameters that have worked well previously. The
combination of the surrogate model and acquisition function determines how
well the model performs. Different choices of parameters and functions
can radically change Bayesian optimization from quickly converging to the
best value to being worse than random sampling. The only way to determine
the best Bayes opt parameters is empirically.

Tutorials:

For the mathematically minded this paper provides a good summary of Bayesian
optimization:

https://arxiv.org/pdf/1807.02811.pdf

For those less mathematically inclined

https://towardsdatascience.com/the-beauty-of-bayesian-optimization-explained-in-simple-terms-81f3ee13b10f

## Contributors

The initial implementation/repository design was done by Austin Talbot
(firstname . lastname 1993 @ gmail). Corey Keller has the final say on
any decisions regarding this repository.

# Dependencies

The dependencies are listed in requirements\_botorch.txt, but listed here
they are
1. jupyter - allows for Jupyter notebooks to demonstrate functioning and store results
2. gym - allows for interactive simulations
3. tqdm - trange allows for progress measurement in for loops
4. botorch - Facebook's Bayesian optimization implementation
5. matplotlib - visualization library analagous to MATLAB (MATplotlib)
6. torch - pytorch, basis of botorch
7. torchvision - potentially unnecessary
8. torchaudio - potentially unnecessary

# Setup instructions

Make sure anaconda is downloaded, with instructions found at

https://www.anaconda.com/

Having installed Anaconda, you can create a conda environment to use the
scripts. This can be done in terminal using the following commands:

```
conda create -n bayes_opt python=3.10
conda activate bayes_opt
pip install -r requirements_botorch.txt
```

Having installed the necessary packages you can now use this environment
whenever you want to run the code. Type
```
conda activate bayes_opt
```
to open this environment and type
```
conda deactivate
```
once you have concluded your work. This will prevent you from accidentally
installing extra packages in the environment that can potentially break the
implementation.

# Components

Code - this directory has the Bayesian optimization code. At this moment
the only implementation uses BoTorch.

Demos - this directory has a series of examples, demonstrating how the code
can be used. This will be similar to the scripts in UnitTests, except it
will be nicely formatted so that anyone can understand how to use these
scripts with minimal python experience.

UnitTests - this directory has a list of unit tests to ensure proper
functioning of the code. These are example scripts where the behavior is
known by default to ensure proper functioning of any code
additions/modifications.

# Version control and development practices

Every script in UnitTests/SmokeTest should be run before commiting to the
main repository. This will ensure that the basic functions compile and that
changes do not affect basic structure. This is simple to do on MACOS or
Linux! In Terminal simply type the commands
```
chmod 777 runAll.sh
./runAll.sh
```
After you do this once the first line is unnecessary! There is simply no
excuse for not running this every time before a commit. It will print a
bunch of text and if
```
!!!!!!!!!!!
!!!!!!!!!!!
Test Failed
!!!!!!!!!!!
!!!!!!!!!!!
```
appears anywhere in the output figure out what's wrong before commiting!

Every commit should include the
version number (which should be incremented, and the associated string
*version* included in the code files changed accordingly). Therefore, when
any saved model is loaded it is straightforward to see which version of
the code that model used.

# Code documentation

The code should be ***well-commented***, meaning that the comments in the
code should be informative enough that anyone even without a technical
background should be able to understand how the code works by the comments
alone. The first lines in every file should be a triple-quoted string that
documents the file so that anyone can type
```
import filename
print(filename.__doc__)
```
and have the contents of that file summarized. It should include a
paragraph(s) summary, a list of any objects, methods, a contributor list,
and a version history.

Any function should also have its first line be a triple-quoted string so
that the lines of code
```
from filename import func
from filename import ExampleObject
myObject = ExampleObject(params)
print(func.__doc__)
print(myObject.method.__doc__)
```
provide information on what the method does. It should provide a line
summary of the method purpose, a list of parameters (if any), and what the
method returns. All of these should have type and default value (if any)
included.

Any object methods that are not meant to be used by a biologist should
begin with an underscore.

Every single object should have the \_\_repr\_\_ implemented so that the
line
```
print(myObject)
```
should actually provide *meaningful* information to the user. At a minimum
it should provide the version number and creation date of the object.




