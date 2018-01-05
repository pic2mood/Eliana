# Eliana [![Build Status](https://travis-ci.org/raymelon/Eliana.svg)](https://travis-ci.org/raymelon/Eliana) ![](https://reposs.herokuapp.com/?path=raymelon/Eliana)

**Eliana predicts the human emotion response towards a presented image.**

## What Eliana?

**Eliana is an implementation of the Object-to-Image Association (OIA) model, which adds object annotations as features to consider in prediction. Uses [MSCOCO annotated objects (Lin et. al., 2015)](http://arxiv.org/abs/1405.0312), [colorfulness score (Hasler and  Susstrunk, 2003)](https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf), [dominant colors palette (using color-thief)](https://github.com/fengsp/color-thief-py), and [Mean GLCM contrast/texture (Haralick et. al., 1973)](https://www.mathworks.com/help/images/texture-analysis-using-the-gray-level-co-occurrence-matrix-glcm.html?requestedDomain=www.mathworks.com) features.**

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

   - **Default (without [OIA model](#what-eliana))**
      ```Bash
      make run
      ```
      
   - **[OIA](#what-eliana) model**
      ```Bash
      make run mode=oia
      ```

#### To train
   - **Default (without [OIA model](#what-eliana))**
      ```Bash
      make train
      ```
      
   - **[OIA](#what-eliana) model**
      ```Bash
      make train mode=oia
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
- Commiting comes in the following format:
   1. Format Type 1 (for repo handlers)
       
       ```
       [<TYPE OF COMMIT>] v<MAJOR>.<MINOR>.<PATCH> build <TRAVIS-BUILD> (#<ISSUE>) <COMMIT MESSAGE>`
       ```

   2. Format Type 2 (for pull requests)
   
      ```
      [<TYPE OF COMMIT>] (#<ISSUE>) <COMMIT MESSAGE>
      ```

- The types of commits are as follows:


     | TYPE | USE FOR |
     |----- | ------- |
     | FIX | Bug fixes |
     | SETUP | New technology setups for project's use |
     | TWEAK | Enhancement of project components for performance |
     | REFACTOR | Enchancement of project components for style |
     | FEATURE | New features |
     | DOC | Any documentation stuff |

- The project uses [Semantic Versioning 2.0.0](http://semver.org/) for versioning.

- Build numbers `<TRAVIS-BUILD>` follows the [project's Travis-CI build count](https://travis-ci.org/raymelon/Eliana). 

- Note that pull requests aren't built automatically, and thus for this type of commit it is advised to use the Format Type 2.



