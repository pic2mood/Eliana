# Eliana [![Build Status](https://travis-ci.org/raymelon/Eliana.svg)](https://travis-ci.org/raymelon/Eliana) ![](https://reposs.herokuapp.com/?path=raymelon/Eliana)
Eliana is a prediction engine aiming at predicting the most likely human emotion response towards an image of different photographic genres.

## Why Eliana?

Just a beautiful random name for our engine. The name itself derives from Hebrew, meaning *"My God has answered me."* Thinking of a name is something that drains your brain out, and yes, God has answered our plea for a beautiful name.

## Contributing Guidelines
### I. Setup
1. Fork the project.
2. Clone your own/forked copy of the project to your local machine.
3. Install [Anaconda](https://www.anaconda.com/downloads).
4. Download the conda enviroment.

   - [Packed  (MEGA)](https://mega.nz/#!82gBBCqT!clt5iihZZGYDGOE6utsr207iNviRAFqbI-_TsPFmswQ)
   - [Main (MEGA)](https://mega.nz/#F!Yn4WzY6I!3o2klQ-LfVwkTt61yVA9Gw)
   - [Mirror (Google Drive)](https://drive.google.com/open?id=0B2Gw0zD3SerkVWtsSVlRTUNuWVE)
   
5. Extract the downloaded env.   
6. To ensure an updated env, sync the extracted env through Link 2.
7. Delete `env` on the project folder.
8. Copy the extracted env to the project folder.
9. Activate the copied env.
   ```Bash
   source activate ./env/eliana
   ```
10. Run `unit_test_all.py` and `integrated.py` to make sure all is working fine.
   ```Bash
   python -m tests.unit_test_all.py && python -m integrated.py
   ```

### II. Commiting
- Commiting comes in the following format:
   1. Format Type 1 (for repo handlers)
       
       ```
       [<TYPE OF COMMIT>] v<MAJOR>.<MINOR>.<PATCH> build <TRAVIS-BUILD>`
       ```

   2. Format Type 2 (for pull requests)
   
      ```
      [<TYPE OF COMMIT>]
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

## Changelog
See [CHANGELOG.md](https://github.com/raymelon/Eliana/blob/0.1-pre/CHANGELOG.md).



