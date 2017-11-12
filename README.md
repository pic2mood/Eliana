# Eliana [![Build Status](https://travis-ci.org/raymelon/Eliana.svg)](https://travis-ci.org/raymelon/Eliana) ![](https://reposs.herokuapp.com/?path=raymelon/Eliana)
Eliana is a prediction engine aiming at predicting the most likely human emotion response towards an image of different photographic genres.

## Why Eliana?

Just a beautiful random name for our engine. The name itself derives from Hebrew, meaning *"My God has answered me."* Thinking of a name is something that drains your brain out, and yes, God has answered our plea for a beautiful name.

## Contributing Guidelines
### I. Setup
1. Fork the project.
2. Clone your own/forked copy of the project to your local machine.
2. Download the conda enviroment.

   - [Link 1 (Archived as .tar.xz)](https://mega.nz/#!82gBBCqT!clt5iihZZGYDGOE6utsr207iNviRAFqbI-_TsPFmswQ) at MEGA
   
   - [Link 2](https://drive.google.com/open?id=0B2Gw0zD3SerkVWtsSVlRTUNuWVE) at Google Drive
   
3. Extract the downloaded env.   
4. To ensure an updated env, sync the extracted env through Link 2.
5. Delete `env` on the project folder.
6. Copy the extracted env to the project folder.
8. Activate the copied env.
   ```Bash
   source activate ./env/eliana
   ```
7. Run `unit_test_all.py` and `integrated.py` to make sure all is working fine.
   ```Bash
   python -m tests.unit_test.all.py && python -m integrated.py
   ```

### II. Commiting
- Commiting comes in the following format:
   1. Format Type 1 (for repo handlers)
       
       ```
       [<TYPE OF COMMIT>] (v<MAJOR>.<MINOR>.<PATCH>) build <TRAVIS-BUILD>`
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

- The project uses [Semantic Versioning 2.0.0](http://semver.org/) for versioning.

- Build numbers `<TRAVIS-BUILD>` follows the [project's Travis-CI build count](https://travis-ci.org/raymelon/Eliana). 

- Note that pull requests aren't built automatically, and thus for this type of commit it is advised to use the Format Type 2.




