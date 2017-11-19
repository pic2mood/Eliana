# Eliana Changelog




### [0.1.31-pre build 42 (, 11-17-17)]()
[REPO](https://github.com/raymelon/Eliana/blob/0.1-pre/)
- **[REFACTOR]** Made the repository's directory structure compatible for Python packaging.
- **[DOCS]** Rebuilt docs for new directory structure.


### [0.1.30-pre build 41 (#6db832c, 11-17-17)](https://github.com/raymelon/Eliana/commit/6db832ce0cd278e2513b04987e279375d2f0ef2d)
[scripts/texture_script.py](https://github.com/raymelon/Eliana/blob/0.1-pre/scripts/texture_script.py)
- **[FEATURE]** Initial texture module script.



### [0.1.28-pre build 39 (#390f6a9, 11-15-17)](https://github.com/raymelon/Eliana/commit/390f6a95bf50bf41908bcd063c1b31e3400816f8)

[lib/color/color.py](https://github.com/raymelon/Eliana/blob/0.1-pre/lib/color/color.py)
- **[FEATURE]** Initial color module implementation.

[tests/color_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/color_unit_test.py)
- **[FEATURE]** Initial unit test for the new color module.




### [0.1.26-pre build 37 (#e50a917, 11-14-17)](https://github.com/raymelon/Eliana/commit/e50a91719acfb7fc83c204654054bfdc8946cd9e)

[lib/annotator/annotator.py](https://github.com/raymelon/Eliana/blob/0.1-pre/lib/annotator/annotator.py)
- **[REFACTOR]** Minor edits in accordance to ElianaImage new features.

[lib/image/eliana_image.py](https://github.com/raymelon/Eliana/blob/0.1-pre/lib/image/eliana_image.py)
- **[FEATURE]** Added flexibility on constructor by supporting path, np image and pil image arguments.

[tests/annotator_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/annotator_unit_test.py)
- **[REFACTOR]** Minor edits in accordance to ElianaImage new features.




### [0.1.25-pre build 33 (#32fcb32, 11-14-17)](https://github.com/raymelon/Eliana/commit/32fcb32a216e67e9e177929daf02f5cdc7f0d7ef)

[lib/annotator/annotator.py](https://github.com/raymelon/Eliana/blob/0.1-pre/lib/annotator/annotator.py)
- **[REFACTOR]** Applied the new ElianaImage refactors, plus major refactors on the module itself.

[lib/image/eliana_image.py](https://github.com/raymelon/Eliana/blob/0.1-pre/lib/image/eliana_image.py)
- **[REFACTOR]** Renamed to from image.py to eliana_image.py.
- **[REFACTOR]** Moved to from lib/annotator to lib/image.

[tests/annotator_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/annotator_unit_test.py)
- **[REFACTOR]** Applied the new ElianaImage refactors.




### [0.1.24-pre build 32 (#fe24fbb, 11-13-17)](https://github.com/raymelon/Eliana/commit/fe24fbba60f6838853736d9fdb7618c1efaf6d71)

[tests/eliana_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/eliana_test.py)
- **[FEATURE]** Added tester function for function calls.

[tests/annotator_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/annotator_unit_test.py)
- **[FEATURE]** Applied the new tester function.




### [0.1.23-pre build 31 (#0691e5b, 11-13-17)](https://github.com/raymelon/Eliana/commit/0691e5b3a6614b2cca42bf4882c3e1d9571e059a)

[.travis.yml](https://github.com/raymelon/Eliana/blob/0.1-pre/.travis.yml)
- **[SETUP]** Added install commands.




### [0.1.22-pre build 30 (#af3d244, 11-13-17)](https://github.com/raymelon/Eliana/commit/af3d24421a94f11b9e8e5c98cb3dfe65b38c9b77)

[README.md](https://github.com/raymelon/Eliana/blob/0.1-pre/README.md)
- **[DOC]** Added [DOC] type of commit and Changelog section.

[tests/annotator_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/annotator_unit_test.py)
- **[REFACTOR]** Used dir_env_modules in appropriate manner




### [0.1.21-pre build 29 (#eafd294, 11-12-17)](https://github.com/raymelon/Eliana/commit/eafd294ce8042d8545c47e6b34bc99b43bcf8e6a)

[tests/eliana_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/eliana_test.py)
- **[FEATURE]** Initial commit for this file.

[tests/annotator_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/annotator_unit_test.py)
- **[REFACTOR]** Used the classes provided by tests/eliana_test.py to modularize testing.




### [0.1.18-pre build 25 (#89ce49a, 11-12-17)](https://github.com/raymelon/Eliana/commit/89ce49a)

[lib/annotator/annotator.py](https://github.com/raymelon/Eliana/blob/0.1-pre/lib/annotator/annotator.py)
- **[DOC]** Add docstrings on all functions and members.

[lib/tests/annotator_unit_test.py](https://github.com/raymelon/Eliana/blob/0.1-pre/tests/annotator_unit_test.py)
- **[REFACTOR]** Changed import dependencies from adding lib paths to syspath to relative imports.



