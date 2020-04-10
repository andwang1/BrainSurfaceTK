# GUI

For anyone who can be asked to read this. 

The main server has been developed using the Django framework. Please note that the folder **main** actually contains all the main project code & static files. **BasicSite** acts as the head of the server. 

Have set up mod_wsgi to serve static files.
See documentation at: [https://pypi.org/project/mod-wsgi/](url)

To set up the modwsgi, please run:
`mod_wsgi-express setup-server BasicSite/wsgi.py --port=8080 --user www-data --group www-data --server-root={ABSOLUTE PATH TO PARENT OF *BasicSite*}`

To run the server, use: 
`python manage.py runmodwsgi`

Package-wise:

*  Nibabel is used to load the raw MRI patient files
*  Nilearn is used to output an interactive slicing tool for the loaded MRI patient files.
*  VTK.js is used to load & interactively display .vtp brain surface files.


TODO:
*  Setup Apache or Nginx for static file serving or try to find a work around.
*  Optionally, could set up patient data upload, with form.


