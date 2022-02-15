## Vis4D Documentation

**Note** 
- All images and videos should go to img/ folder
- The documentation will be automatically deployed to the website once a PR is merged to main


### Build

Install dependencies

```
pip install -r requirements.txt
```

Build the documentation

```
make html
```
_Please make sure the documentation can build correctly before merging changes to main!_

The full doc files will be in `build` and can be displayed in the browser by opening `build/html/index.html`.


#### Notes

The initial API documentation was created with:

```
sphinx-apidoc -E -f -o ./source/api/ ../vis4d/ ../vis4d/unittest/* ../vis4d/**/*_test.py
```

The API documentation will be updated automatically via sphinx-autodoc, so there is no need to run this command during a PR.
