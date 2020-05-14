# GUI
This server has been developed using the [Django](https://www.djangoproject.com/) framework. We use [MOD-WSGI](https://modwsgi.readthedocs.io/en/develop/) Standalone to run an Apache server to host this webapp.
The MRI visualisation is done thanks to [Nilearn](https://nilearn.github.io/index.html) and the Brain surface is displayed using [VTK.js](https://kitware.github.io/vtk-js/index.html).
###### Run instructions
After following the instructions on creating a virtual environment containing all of our dependencies:
1. First you will want to create a super user. This can be done by: 
```
chmod 700 ./createsuperuser.sh
./createsuperuser.sh
```
2. Next you may want to use your own original data, this can be done by overwriting the meta_data.tsv data file in ``GUI/media/original/data``, please take care that the column names are exactly the same and in the same order. If not, then the load data function that can be called in the admin panel will not work. 
3. After you've created a super user, you can either run the server in developement mode by running:
```
python startserver.py
```
4. Alternatively you may wish to run the server in production mode. If you want others to remotely access this server, you may need to open port 8000 on your machine and please don't forget to port forward if you are using a modem. After you have done this, you can simply run:
```
python startserver.py prod
```

###### Settings
Within the ``GUI/BasicSite/setting.py``, you should add the IP address and if allowed DNS ``ALLOWED_HOSTS``.
If you want to change the original directory, you'll have to change the path of the media path and specify where the original data is.
