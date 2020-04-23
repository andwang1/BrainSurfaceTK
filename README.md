# GUI

For anyone who can be asked to read this. 

The main server has been developed using the Django framework. Please note that the folder **main** actually contains all the main project code & static files. **BasicSite** acts as the head of the server. 

Have set up mod_wsgi to serve static files.
See documentation at: [https://pypi.org/project/mod-wsgi/](https://pypi.org/project/mod-wsgi/)

To set up the modwsgi, please run: 
`python -m pip install mod-wsgi`

To run the server, use: 
`python manage.py runmodwsgi --url-alias /media media`

Currently it appears that we have a package conflict between PyVista & Mod-WSGI. This is problematic since both are crucial.

Package-wise:

*  Nibabel is used to load the raw MRI patient files
*  Nilearn is used to output an interactive slicing tool for the loaded MRI patient files.
*  VTK.js is used to load & interactively display .vtp brain surface files.


TODO:
*  Setup Apache or Nginx for static file serving or try to find a work around.
*  Optionally, could set up patient data upload, with form.


