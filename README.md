# Eliana [![Build Status](https://travis-ci.org/raymelon/Eliana.svg)](https://travis-ci.org/raymelon/Eliana) ![](https://reposs.herokuapp.com/?path=raymelon/Eliana)

**Eliana predicts the human emotion response towards a presented image.**

## What Eliana?

**Eliana is an implementation of the Object-to-Emotion Association (OEA) model, which adds object annotations as features to consider in predicting human emotion response towards an image. Uses [MSCOCO annotated objects (Lin et. al., 2015)](http://arxiv.org/abs/1405.0312), [colorfulness score (Hasler and  Susstrunk, 2003)](https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf), [dominant colors palette (using color-thief)](https://github.com/fengsp/color-thief-py), and [Mean GLCM contrast/texture (Haralick et. al., 1973)](http://haralick.org/journals/TexturalFeatures.pdf) features.**

**Built on top of [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).**

- [Anaconda==4.3.30](https://www.anaconda.com/download/)
- Python packages:
```
# Available on both conda and pip
scikit-image==0.13.0
scikit-learn==0.19.1
tensorflow==1.3.0
pillow==3.4.2
pandas==0.20.1
numpy<=1.12.1
opencv==3.1.0

# Available on pip only
colorthief==0.2.1
```
- External APIs:
   - [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Why Eliana?

Just a beautiful random name for our engine. The name itself derives from Hebrew, meaning *"My God has answered me."* Thinking of a name is something that drains your brain out, and yes, God has answered our plea for a beautiful name.

## Contributing Guidelines
### I. Setup
1. **Fork and clone the project.**
2. **Install [Anaconda](https://www.anaconda.com/downloads).**
3. **Setup the virtual enviroment, through**
   1. The packaged virtual environment
        - Download the conda enviroment.
            - [Main (MEGA)](https://mega.nz/#F!Yn4WzY6I!3o2klQ-LfVwkTt61yVA9Gw)
            - [Mirror (Google Drive)](https://drive.google.com/open?id=0B2Gw0zD3SerkVWtsSVlRTUNuWVE)
            - [Packed  (MEGA)](https://mega.nz/#!82gBBCqT!clt5iihZZGYDGOE6utsr207iNviRAFqbI-_TsPFmswQ)
            
               ***This one needs to be synced to the Main link*
             
      or,          
   2. Manual setup of requirements

4. **Change directory to the virtual environment directory, then activate it.**
   ```Bash
   source activate ./env/eliana
   ```
   
### II. Builds, run and tests

#### Running the executable module

   - **Default (with [OEA](#what-eliana) model)**
      ```Bash
      make run # or make run model=oea
      ```
      
   - **[OEA](#what-eliana)-less model**
      ```Bash
      make run model=oea_less
      ```

#### To train
   - **Default (with [OEA](#what-eliana) model)**
      ```Bash
      make train # or make train model=oea
      ```
      
   - **[OEA](#what-eliana)-less model**
      ```Bash
      make train model=oea_less
      ```

#### Running tests
```Bash
make test
```

#### Build documentation
```Bash
make doc
```

### III. Commiting
#### Commit messages comes in the following format:

   1. **Format Type 1 (for main repos)**
       
       ```bash
       git commit -m "[{commit type}] v{major}.{minor}.{patch} b{travis build no.}. (#{tracker no.}) (#{issue no.}) {commit message}."
       ```
       Example:
       ```bash
       # with issue
       git commit -m "[doc] v0.1.1 b1. (#24) Changes on commit message format."
       
       # multiple issues
       git commit -m "[doc] v0.1.1 b1. (#24 #32) Changes on commit message format."
       
       # with tracker
       git commit -m "[doc] v0.1.1 b1. (#8) (#24 #32) Changes on commit message format."
       ```

   2. **Format Type 2 (for forks)**
   
      ```bash
      git commit -m "[{commit type}] (#{issue no.}) (#{tracker no.}) {commit message}"
      ```
      Example:
       ```bash
       git commit -m "[doc] (#24) Changes on commit message format."
       ```

#### The types of commit messages are as follows:

   | TYPE | USE FOR | TRACKERS |
   |----- | ------- | -------- |
   | **doc** | Documentation stuff (README, LICENSE, Sphinx doc) | Sphinx ([#8](https://github.com/raymelon/Eliana/issues/8)), README ([#42](https://github.com/raymelon/Eliana/issues/42)) 
   |         | `[doc] v0.1.1 b1. (#8) (#24) Changes on commit message format.` |
   | **feature** | New feature |
   |             | `[feature] v0.1.1 b1. Added color filter on colorfulness module.` |
   | **fix** | Bug fixes |
   |         | `[fix] v0.1.1 b1. (#6) Fix on non-showing image.` |
   | **merge** | Pull request merges |
   |           | `[merge] v0.1.1 b1. Merge pull request #40 from raymelon/0.1-pre.` |
   | **refactor** | Style/format enhancements |
   |           | `[refactor] v0.1.1 b1. Made spacing in accordance to PEP8.` |
   | **setup** | Setup of new third-party technology/system |
   |           | `[setup] v0.1.1 b1. Create .travis.yml` |
   | **tweak** | Performance enchancements |
   |           | `[tweak] v0.1.1 b1. OEA model accuracy tuning.` |
     

- The project uses [Semantic Versioning 2.0.0](http://semver.org/) for versioning.

- Patch versions are updated every commit.

- Build numbers `<TRAVIS-BUILD>` follows the [project's Travis-CI build count](https://travis-ci.org/raymelon/Eliana). 

- Build numbers are updated every push.

- Note that pull requests aren't built automatically, and thus for this type of commit it is advised to use the Format Type 2.



