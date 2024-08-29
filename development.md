## Development of the hypso-package

## Development of the pip package
- [Packaging projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- Create an account at [PyPI.org](https:/pypi.org) and request access to the hypso project (contact Cameron or Sivert, updated 2024-08-29)
- Add your [PyPI.org](https:/pypi.org) login credentials and token to `~/.pypirc`
- Install the `setuptools` build system and `twine` using pip 
- Update the version number in `pyproject.toml`
    - Use the version number format "vX.Y.Z" for normal releases 
    - Use the version number format "vX.Y.Z.a1" for alpha releases 
    - Use the version number format "vX.Y.Z.b1" for beta releases 
- Build the package with `python3 -m build`
- Upload the newly built package to PyPI: `python3 -m twine upload --repository pypi dist/*`
- View the project at [pypi.org/project/hypso/](https://pypi.org/project/hypso/)

## Development of the conda-forge package

There are two main repos:
- Package: https://github.com/NTNU-SmallSat-Lab/hypso-package
- Conda-Forge Feedstock: https://github.com/conda-forge/hypso-feedstock

You need to:
- Create a Fork of the hypso-package to implement new functionality
- Create Fork of hypso-feedstock to trigger a conda-forge upversion

```diff
- IMPORTANT: Only modify your forks and create a PR to merge. Never modify directly.

```

### To Update conda-forge Package


#### 1. Modify your fork of the hypso package with new improvements
Do not forget to update the `pyproject.toml` file in the root. The `fallback_version` should match with the next step bundle name.
![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/bab5072e-cecb-4973-888a-26238c95a3ec)

If you added new packages include them in the same file (see image below). Versions of the files may be required.

![Screenshot 2024-03-17 at 03 40 33](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/f61eda77-2830-4956-a7a8-711b5085007b)


#### 2. Bundle a Sub-Sequent Release
You need to create a new release. The recommended format is 3 digits with a lowecase "v" as a prefix. Example: v1.9.9 This number should be the same fone as in Step 1. Create a "tag" when creating a release version or Step 3 wont work.

![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/6b920b92-6301-447a-860a-9c11720c2923)


![Screenshot 2024-03-17 at 03 36 36](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/3e43d2ef-b464-497c-bad7-a1708e6554a3)

#### 3. Generate SHA 256 for the newly released file

Use the following code and change the version relesed (for linux and mac). Modify the version of the like to match step 1 and 2.

    curl -sL https://github.com/NTNU-SmallSat-Lab/hypso-package/archive/v1.9.9.tar.gz | openssl sha256
    
Copy the SHA256 string that you get after running that code above for the next step.

#### 4. Modify your fork of hypso-feedstock

On your fork, modify the file `recipe/meta.yaml`, specifically *lines 2 and 11* in the following image:

![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/4dea09f0-009e-4789-98da-6c7a706721c4)

The version should match all previous steps and the sha256 value should be the one generated in Step 3.

If new packages are added they should be included in the *"run"* section with the same name and version as written in the `pyproject.toml` file (see Step 1).

![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e144b135-b42d-4418-b1b1-3e7944675953)


#### 5. Create a Pull Request (PR) for your hypso-feedstock fork
Complete the checklist in the PR template

![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/78fdb5e2-b057-42a3-9fb0-9d331b8d93a1)

Bots will check the PR and the file will be compiled on Azure to make sure everything works. If an error occurs you will see it. Once all the tests are passed, the PR can be merged.

![Screenshot 2024-03-17 at 03 53 37](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/e3c96101-e73a-4e98-a66c-f101f82b7b9e)

After the PR is merged, the new version will be available for install in conda-forge.

![image](https://github.com/NTNU-SmallSat-Lab/hypso-package/assets/87340855/3b1ce72b-bcc0-4b74-8257-165216ab291f)


## Important Considerations

1. Importing files needs to be done using package file imports like the following line of code.

```
full_rad_coeff_file = files('hypso.calibration').joinpath(
                f'data/{"radiometric_calibration_matrix_HYPSO-1_full_v1.csv"}')
```
    
2. Any non-python file that wants to be included to be uploaded needs to be added in the `MANIFEST.in` file
3. Packages names and version in both the `pyproject.toml` and `meta.yaml` are case and space sensitive, be carefull with the spacing. Avoid using specific versions (==) and try to use higher than (>=) as it makes it easier for future compatibility.
